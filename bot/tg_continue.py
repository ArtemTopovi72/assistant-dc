"""▶️ Continue video (Creativity > Video): the user's clip goes on from where it ended.

States on the session (continue_state), same shape as 🎨 Restyle:
  "want_video"  the button was pressed; the next video / round video / video file is the clip.
  "want_text"   the clip is saved; the next plain text is what should happen next.

How the continuation is built (50 searches, 2026-10-04, docs/video_continuation_sota_2026-10.md):
  * the last ~4 s of the clip ride as <Video 1>: its motion, camera speed, faces AND its own
    soundtrack (the voices) -- one still frame fixes the picture but not its direction, and the
    clip stalls at the join; a reference video carries its sound with it;
  * one frame rides as <Picture 1>: the sharpest of the last 0.5 s. Not further back: live
    10-06 the "sharpest moving frame of the last 2 s" was 2 s before the end, the coat taken
    off in those 2 s was back on in the continuation;
  * generate_video wraps the user's words in a «continue, no cut» template and, when the new
    part is done, joins original + continuation with a crossfade (media/video.join_continuation).
"""
import os
import subprocess
import uuid

TAIL_SECONDS = 2.5          # the reference clip: its motion and its sound (longer = older states)
FRAME_WINDOW = 3.0          # seconds looked at for the start frame
FRAME_CANDIDATES = 5        # ... of which the last 0.5 s (at 10 fps) can be chosen
MAX_PEOPLE = 2               # new people from photos (<Picture 2>, <Picture 3>) a continuation takes


def seed_frame(src: str, out_jpg: str) -> bool:
    """Save the start frame for the continuation as out_jpg: the sharpest of the last 0.5 s
    (preferring frames still moving as fast as the clip was -- a blurred swing makes a bad
    first frame). Never earlier: whatever happened in the clip must stay happened."""
    import numpy as np
    from PIL import Image
    import video_look
    tmp = out_jpg + ".tail"
    os.makedirs(tmp, exist_ok=True)
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-sseof", f"-{FRAME_WINDOW}", "-i", src,
                        "-vf", "fps=10", "-q:v", "2", os.path.join(tmp, "f_%03d.jpg")],
                       capture_output=True, check=True, timeout=60)
        frames = sorted(os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".jpg"))
        if not frames:
            return False
        gray = [np.asarray(Image.open(p).convert("L").resize((320, 320)), dtype=np.float32) for p in frames]
        sharp = [video_look._sharpness(g) for g in gray]
        motion = [0.0] + [float(np.abs(gray[i] - gray[i - 1]).mean()) for i in range(1, len(gray))]
        typical = float(np.median(motion[1:])) if len(motion) > 1 else 0.0
        pool = list(range(max(0, len(frames) - FRAME_CANDIDATES), len(frames)))
        moving = [i for i in pool if motion[i] >= 0.6 * typical]      # a still clip: everything counts
        best = max(moving or pool, key=lambda i: sharp[i])
        Image.open(frames[best]).convert("RGB").save(out_jpg, quality=95)
        return True
    except Exception:
        return False
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def cut_tail(src: str, out_mp4: str, seconds: float = TAIL_SECONDS) -> bool:
    """The last `seconds` of the clip, with its sound, as the reference video."""
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-sseof", f"-{seconds}", "-i", src,
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "160k", out_mp4],
                       capture_output=True, check=True, timeout=120)
        return os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 0
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
        people = list(getattr(sess, "continue_people", None) or [])
        if len(people) >= MAX_PEOPLE:
            self._send_text(chat_id, tg_bot._t("continue_people_full", lang))
            return True
        data = self._dl_bytes(fid)
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

    def _continue_take_media(self, chat_id: int, sess, lang: str, msg: dict) -> bool:
        if getattr(sess, "continue_state", "") not in ("want_video", "want_text"):
            return False
        fid = self._video_of(msg)
        if not fid:
            return self._continue_take_person(chat_id, sess, lang, msg)
        data = self._dl_bytes(fid)
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
        if not (seed_frame(src, frame) and cut_tail(src, tail)):
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
        sess.continue_state = ""         # disarmed before the slow part
        self._store.put(sess)
        # the ordinary animate request; tg_tasks hands the clip's tail to generate_video
        self._enqueue_item(chat_id, {"type": "text", "text": "animate this photo: " + text.strip()})
        return True


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)
