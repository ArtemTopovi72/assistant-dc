"""▶️ Continue video (Creativity > Video): the user's clip goes on from where it ended.

States on the session (continue_state), same shape as 🎨 Restyle:
  "want_video"  the button was pressed; the next video / round video / video file is the clip.
  "want_text"   the clip is saved; the next plain text is what should happen next.

How the continuation is built (see docs/video_continuation_sota_2026-10.md):
  * the last ~2.5 s of the clip ride as <Video 1>: its motion, camera speed, faces AND its own
    soundtrack (the voices) -- one still frame fixes the picture but not its direction, and the
    clip stalls at the join; a reference video carries its sound with it;
  * the final frame rides as <Picture 1>, preserving changes made at the end of the clip;
  * generate_video wraps the user's words in a «continue, no cut» template and, when the new
    part is done, joins original + continuation with a crossfade (media/video.join_continuation).
"""
import os
import subprocess
import uuid
from seams import dep

TAIL_SECONDS = 2.5          # the reference clip: its motion and its sound (longer = older states)
MIN_REFERENCE_SECONDS = 2.0  # MiniMax H3 reference clips must be at least two seconds
FRAME_WINDOW = 3.0          # seconds looked at for the start frame
MAX_PEOPLE = 2               # new people from photos (<Picture 2>, <Picture 3>) a continuation takes


def _video_tail_seek(src: str, seconds: float):
    from video import probe
    duration = probe(src).get('video_seconds') or 0
    if not duration:
        try:
            import json
            result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                     '-show_entries', 'stream_tags=DURATION', '-of', 'json', src],
                                    capture_output=True, text=True, check=True, timeout=60)
            tag = json.loads(result.stdout)['streams'][0]['tags']['DURATION']
            hours, minutes, secs = tag.split(':')
            duration = int(hours) * 3600 + int(minutes) * 60 + float(secs)
        except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError):
            pass
    if duration > 0:
        return ['-ss', str(max(0.0, duration - seconds))], min(duration, seconds)
    return ['-sseof', f'-{seconds}'], None


def seed_frame(src: str, out_jpg: str) -> bool:
    """Save the clip's actual last video frame as the continuation's start frame."""
    import tempfile
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix='.jpg', dir=os.path.dirname(os.path.abspath(out_jpg)))
        os.close(fd)
        seek, _ = _video_tail_seek(src, FRAME_WINDOW)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *seek, "-i", src,
                        "-an", "-fps_mode", "passthrough", "-q:v", "2", "-update", "1", tmp],
                       capture_output=True, check=True, timeout=60)
        if not os.path.getsize(tmp):
            return False
        os.replace(tmp, out_jpg)
        return True
    except Exception:
        return False
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass


def face_frame(src: str, out_jpg: str, circle: bool = False) -> bool:
    """The frame of a clip that shows its person best -- the largest, sharpest face over the
    whole clip -- as out_jpg; `circle` crops a кружок to the square inside its disc first.
    A person is a look, not a moment: their LAST frame (hand over the mouth, blurred) gave
    the video model nothing to go on, and a stranger walked out (live 2026-10-12)."""
    import numpy as np
    from PIL import Image
    import identity_metrics
    import video_look
    tmp = out_jpg + ".all"
    os.makedirs(tmp, exist_ok=True)
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", "fps=3", "-q:v", "2",
                        os.path.join(tmp, "f_%03d.jpg")], capture_output=True, check=True, timeout=120)
        frames = sorted(os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".jpg"))
        if not frames:
            return False
        scored = []
        for p in frames:
            with Image.open(p) as im:
                im = im.convert("RGB")
                if circle:
                    w, h = im.size
                    k = 0.707 / 2
                    im = im.crop((int(w * (0.5 - k)), int(h * (0.5 - k)), int(w * (0.5 + k)), int(h * (0.5 + k))))
                    im.save(p, quality=95)
            boxes = identity_metrics.face_boxes(p, pad=0.0)
            if not boxes:
                continue
            x0, y0, x1, y1 = boxes[0]
            g = np.asarray(Image.open(p).convert("L").crop((x0, y0, x1, y1)).resize((160, 160)), dtype=np.float32)
            scored.append(((x1 - x0) * (y1 - y0) * (1.0 + video_look._sharpness(g)), frames.index(p), p))
        if not scored:
            return False
        scored.sort(reverse=True)
        best = scored[0]
        # A second view at least 1.5 s away: a hand over the mouth in one moment is off
        # it in another, and two angles pin a face that one blurry frame does not.
        other = next((s for s in scored[1:] if abs(s[1] - best[1]) >= 5), None)
        if other is None:
            Image.open(best[2]).convert("RGB").save(out_jpg, quality=95)
            return True
        a, b = (Image.open(s[2]).convert("RGB") for s in sorted((best, other), key=lambda s: s[1]))
        b = b.resize((int(b.width * a.height / b.height), a.height))
        sheet = Image.new("RGB", (a.width + b.width, a.height))
        sheet.paste(a, (0, 0))
        sheet.paste(b, (a.width, 0))
        sheet.save(out_jpg, quality=95)
        return True
    except Exception:
        return False
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def first_frame(src: str, out_jpg: str, skip: int = 0) -> bool:
    """The clip's first frame (after `skip` frames) as a picture (the other side of a join from seed_frame)."""
    try:
        import cv2
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip)
        ok, img = cap.read()
        cap.release()
        return bool(ok) and bool(cv2.imwrite(out_jpg, img))
    except Exception:
        return False


def cut_tail(src: str, out_mp4: str, seconds: float = TAIL_SECONDS) -> bool:
    """The last `seconds` of the clip, with its sound, as the reference video."""
    try:
        from video import probe
        seek, duration = _video_tail_seek(src, seconds)
        padding = max(0.0, MIN_REFERENCE_SECONDS - duration) if duration is not None else 0.0
        limit = ['-t', str(duration + padding)] if duration is not None else []
        filters = []
        if padding:
            filters += ['-vf', f'tpad=stop_mode=clone:stop_duration={padding}']
            if probe(src).get('has_audio'):
                filters += ['-af', f'atrim=duration={duration},asetpts=PTS-STARTPTS,apad=pad_dur={padding}']
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *seek, "-i", src, *limit, *filters,
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "160k", out_mp4],
                       capture_output=True, check=True, timeout=120)
        return os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 0 and bool(probe(out_mp4).get('width'))
    except Exception:
        return False


class ContinueMixin:
    def _continue_dir(self, chat_id: int) -> str:
        d = os.path.join(os.path.dirname(str(tg_bot._MASHUP_DIR)), "continue_video", str(chat_id))
        os.makedirs(d, exist_ok=True)
        return d

    def _start_continue_flow(self, chat_id: int, sess, lang: str) -> None:
        sess.continue_state = "want_video"
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("continue_ask_video", lang), parse_mode="HTML")

    def _continue_disarm(self, sess) -> None:
        if getattr(sess, "continue_state", ""):
            sess.continue_state = ""
            self._store.put(sess)

    def _continue_take_person(self, chat_id: int, sess, lang: str, msg: dict) -> bool:
        """A photo after the clip is a NEW person for the continuation (<Picture 2>, <Picture 3>):
        before this it replaced the start frame and the clip went on from the wrong picture."""
        if getattr(sess, "continue_state", "") != "want_text":
            return False
        from tg_dispatch import _largest_photo
        ph = _largest_photo(msg.get("photo"))
        doc = msg.get("document") or {}
        fid = (ph or {}).get("file_id") or (doc.get("file_id") if (doc.get("mime_type") or "").startswith("image/") else None)
        if not fid:
            return False
        generation = msg.get("_context_generation", getattr(sess, "context_generation", 0))
        flow_revision = getattr(sess, "flow_revision", lambda _: 0)
        revision = msg.get("_flow_revision", flow_revision("continue_state"))
        source = getattr(sess, "continue_src", "")
        people = list(getattr(sess, "continue_people", None) or [])
        if len(people) >= MAX_PEOPLE:
            self._send_text(chat_id, tg_bot._t("continue_people_full", lang))
            return True
        data = self._dl_bytes(fid)
        if (generation != getattr(sess, "context_generation", 0) or sess.continue_state != "want_text"
                or revision != flow_revision("continue_state")
                or source != getattr(sess, "continue_src", "")):
            return True
        if not data:
            self._send_text(chat_id, tg_bot._t("continue_fail_dl", lang))
            return True
        path = os.path.join(self._continue_dir(chat_id), f"person_{uuid.uuid4().hex[:8]}.jpg")
        with open(path, "wb") as fh:
            fh.write(data)
        people.append(path)
        sess.continue_people = people
        self._store.put(sess)
        caption = (msg.get("caption") or "").strip()
        if caption:                               # photo + "he walks in and says…": go
            return self._continue_take_text(chat_id, sess, lang, caption)
        self._send_text(chat_id, tg_bot._t("continue_got_person", lang, n=len(people)))
        return True

    def _continue_take_person_clip(self, chat_id: int, sess, lang: str, msg: dict, fid: str) -> bool:
        generation = msg.get("_context_generation", getattr(sess, "context_generation", 0))
        flow_revision = getattr(sess, "flow_revision", lambda _: 0)
        revision = msg.get("_flow_revision", flow_revision("continue_state"))
        source = getattr(sess, "continue_src", "")
        people = list(getattr(sess, "continue_people", None) or [])
        if len(people) >= MAX_PEOPLE:
            self._send_text(chat_id, tg_bot._t("continue_people_full", lang))
            return True
        data = self._dl_bytes(fid)
        if (generation != getattr(sess, "context_generation", 0) or sess.continue_state != "want_text"
                or revision != flow_revision("continue_state")
                or source != getattr(sess, "continue_src", "")):
            return True
        tag = uuid.uuid4().hex[:8]
        clip = os.path.join(self._continue_dir(chat_id), f"person_{tag}.mp4")
        path = clip[:-4] + ".jpg"
        if data:
            with open(clip, "wb") as fh:
                fh.write(data)
        # A кружок frame is a disc with a «TELEGRAM» mark on its rim: given whole, the video
        # model pasted that disc into the corner of every frame (live 2026-10-12).
        ready = data and (dep(self, "face_frame", face_frame)(clip, path, circle=bool(msg.get("video_note")))
                          or dep(self, "seed_frame", seed_frame)(clip, path))
        if (generation != getattr(sess, "context_generation", 0) or sess.continue_state != "want_text"
                or revision != flow_revision("continue_state")
                or source != getattr(sess, "continue_src", "")):
            return True
        if not ready:
            self._send_text(chat_id, tg_bot._t("continue_fail_dl", lang))
            return True
        people.append(path)
        sess.continue_people = people
        self._store.put(sess)
        caption = (msg.get("caption") or "").strip()
        if caption:
            return self._continue_take_text(chat_id, sess, lang, caption)
        self._send_text(chat_id, tg_bot._t("continue_got_person", lang, n=len(people)))
        return True

    def _continue_take_media(self, chat_id: int, sess, lang: str, msg: dict) -> bool:
        state = getattr(sess, "continue_state", "")
        if state not in ("want_video", "want_text"):
            return False
        fid = self._video_of(msg)
        if not fid:
            return self._continue_take_person(chat_id, sess, lang, msg)
        if getattr(sess, "continue_state", "") == "want_text":
            # The clip is already taken: a second video is a NEW person, its frame the
            # picture. A кружок of «the guy with the glove» replaced the grandma's GIF and
            # the continuation went on from the guy, the grandma nowhere (live 2026-10-12).
            return self._continue_take_person_clip(chat_id, sess, lang, msg, fid)
        generation = msg.get("_context_generation", getattr(sess, "context_generation", 0))
        flow_revision = getattr(sess, "flow_revision", lambda _: 0)
        revision = msg.get("_flow_revision", flow_revision("continue_state"))
        data = self._dl_bytes(fid)
        if (generation != getattr(sess, "context_generation", 0) or sess.continue_state != state
                or revision != flow_revision("continue_state")):
            return True
        if not data:
            self._send_text(chat_id, tg_bot._t("continue_fail_dl", lang))
            return True
        tag = uuid.uuid4().hex[:8]
        d = self._continue_dir(chat_id)
        src = os.path.join(d, f"clip_{tag}.mp4")
        with open(src, "wb") as fh:
            fh.write(data)
        frame = os.path.join(d, f"seed_{tag}.jpg")
        tail = os.path.join(d, f"tail_{tag}.mp4")
        ready = dep(self, "seed_frame", seed_frame)(src, frame) and dep(self, "cut_tail", cut_tail)(src, tail)
        if (generation != getattr(sess, "context_generation", 0) or sess.continue_state != state
                or revision != flow_revision("continue_state")):
            return True
        if not ready:
            self._send_text(chat_id, tg_bot._t("continue_fail_dl", lang))
            return True
        # the start frame is the picture the next shot starts from, like an uploaded photo
        sess.target_image = tg_bot._log_image(sess, frame, label=tg_bot._t("to_animate_label", lang), src="user")
        sess.continue_src, sess.continue_tail = src, tail
        sess.continue_people = []                 # a new clip starts a new cast
        sess.continue_state = "want_text"
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("continue_ask_text", lang), parse_mode="HTML")
        return True

    def _continue_take_text(self, chat_id: int, sess, lang: str, text: str) -> bool:
        if getattr(sess, "continue_state", "") != "want_text" or not (text or "").strip():
            return False
        generation = getattr(sess, "context_generation", 0)
        sess.continue_state = ""         # disarmed before the slow part
        flow_revision = getattr(sess, "flow_revision", lambda _: 0)
        revision = flow_revision("continue_state")
        # A submenu armed before the clip (🔎 Поиск) would turn this into "search the
        # web for: animate this photo: …" -- the bot would chat about the
        # search instead of rendering.
        sess.pending_prefix, sess.menu = "", ""
        self._store.put(sess)
        if (generation != getattr(sess, "context_generation", 0) or sess.continue_state
                or revision != flow_revision("continue_state")):
            return True
        # the ordinary animate request; tg_tasks hands the clip's tail to generate_video
        self._enqueue_item(chat_id, {"type": "text", "text": "animate this photo: " + text.strip(),
                                    "_context_generation": generation})
        return True


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)
