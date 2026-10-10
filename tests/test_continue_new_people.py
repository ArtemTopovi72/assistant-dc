"""▶️ Continue video with a NEW person from a photo: the photo sent after the clip is
<Picture 2> (the start frame stays <Picture 1>); before, it replaced the start frame."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "agent", "media", "bot", "voice", "imaging"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")

import tg_continue  # noqa: E402
import video as V  # noqa: E402


class Sess:
    continue_state = "want_text"
    continue_people = []


class Bot(tg_continue.ContinueMixin):
    def __init__(self, d):
        self.d, self.sent, self.queued = d, [], []

        class Store:
            def put(self, s): pass
        self._store = Store()

    def _continue_dir(self, chat_id):
        return self.d

    def _dl_bytes(self, fid):
        return b"\xff\xd8jpeg"

    def _send_text(self, chat_id, text, **kw):
        self.sent.append(text)

    def _enqueue_item(self, chat_id, item):
        self.queued.append(item)

    @staticmethod
    def _video_of(msg):
        return None


def _photo(fid="p1", caption=""):
    m = {"photo": [{"file_id": fid + "s", "width": 90, "height": 90}, {"file_id": fid, "width": 900, "height": 900}]}
    if caption:
        m["caption"] = caption
    return m


def test_a_photo_after_the_clip_is_a_new_person(tmp_path):
    b, s = Bot(str(tmp_path)), Sess()
    s.continue_people = []
    assert b._continue_take_media(1, s, "ru", _photo())
    assert len(s.continue_people) == 1 and os.path.exists(s.continue_people[0])
    assert s.continue_state == "want_text" and not b.queued       # still waits for the text


def test_photo_with_a_caption_starts_the_render(tmp_path):
    b, s = Bot(str(tmp_path)), Sess()
    s.continue_people = []
    assert b._continue_take_media(1, s, "ru", _photo(caption="он входит и здоровается"))
    assert b.queued and "он входит" in b.queued[0]["text"] and s.continue_state == ""


def test_no_more_than_two_people(tmp_path):
    b, s = Bot(str(tmp_path)), Sess()
    s.continue_people = []
    for i in range(3):
        b._continue_take_media(1, s, "ru", _photo(f"p{i}"))
    assert len(s.continue_people) == tg_continue.MAX_PEOPLE


def test_a_photo_before_the_clip_is_not_taken(tmp_path):
    b, s = Bot(str(tmp_path)), Sess()
    s.continue_state, s.continue_people = "want_video", []
    assert not b._continue_take_media(1, s, "ru", _photo())


def test_the_generator_gets_the_start_frame_then_the_people(monkeypatch, tmp_path):
    import tool_image_handlers as H
    seed, person, tail = (tmp_path / "seed.jpg"), (tmp_path / "person.jpg"), (tmp_path / "tail.mp4")
    for p in (seed, person, tail):
        p.write_bytes(b"x")
    seen = {}

    def fake_gen(ctx, description, **kw):
        seen.update(kw, description=description)
        out = tmp_path / "new.mp4"
        out.write_bytes(b"v")
        return {"path": str(out), "status": "success", "seconds": 5.2, "width": 768, "height": 1344}
    monkeypatch.setattr(V, "generate_video", fake_gen)
    monkeypatch.setattr(V, "engine_available", lambda ctx: (True, ""))
    monkeypatch.setattr(V, "join_continuation", lambda a, b: b)
    monkeypatch.setattr(V, "_adopt_output", lambda p: p)
    monkeypatch.setattr(H, "_current_image_paths", lambda ctx, state: [str(seed)])

    class Ctx:
        reference_images = []
        voice_choice = "default"
        continue_tail, continue_src = str(tail), str(tail)
        continue_people = [str(person)]
        def set_stage(self, *_): pass
        def is_cancelled(self): return False
        def remember(self, *a): pass
        def memory_text(self): return ""
    H._handle_generate_video(Ctx(), {}, {"description": "a woman walks in and says «Привет»"})
    assert seen["images"] == [str(seed), str(person)], seen.get("images")
    assert seen["videos"] == [str(tail)]
    assert "<Picture 2>" in seen["description"] and "NEW PEOPLE" in seen["description"]


def test_clause_names_every_new_picture():
    assert "<Picture 2> and <Picture 3>" in V.new_people_clause(2, 2)


def test_continue_text_is_not_a_search_when_a_search_menu_was_open(tmp_path):
    # 🔎 Поиск armed before the clip turned the scene into "search the web for: …"
    b, s = Bot(str(tmp_path)), Sess()
    s.continue_people, s.pending_prefix, s.menu = [], "search the web for: ", "search"
    assert b._continue_take_text(1, s, "ru", "кот приходит и играет с собакой")
    assert s.pending_prefix == "" and s.menu == ""


def test_a_plan_waiting_for_the_user_is_not_a_failed_render():
    # Read as "success with no video" it was retried ~20 times and the turn gave up.
    import graph_personality as G
    msg = "[NOT MADE YET] The script needs several parts, and the user approves the plan first"
    assert G._detect_silent_failure("generate_video", msg, {"video_plan": [{"text": "a"}]}) == msg
    assert G._detect_silent_failure("generate_video", "done", {}).startswith("[TOOL ERROR]")


def test_a_plan_keeps_the_clip_and_the_new_person_for_the_approved_run(monkeypatch, tmp_path):
    # Live 2026-10-10: the plan-only turn used them up; the ▶ run got 1 image, 0 videos.
    import tool_image_handlers as H
    seed, person, tail = (tmp_path / "seed.jpg"), (tmp_path / "cat.jpg"), (tmp_path / "tail.mp4")
    for p in (seed, person, tail):
        p.write_bytes(b"x")
    monkeypatch.setattr(V, "generate_video", lambda *a, **k: (_ for _ in ()).throw(AssertionError("rendered")))
    monkeypatch.setattr(V, "engine_available", lambda ctx: (True, ""))
    monkeypatch.setattr(V, "split_script", lambda text: ["part one " + text, "part two"])
    monkeypatch.setattr(H, "_current_image_paths", lambda ctx, state: [str(seed)])

    class Ctx:
        reference_images = []
        voice_choice = "default"
        video_plan_ask, video_plan_parts = True, []
        continue_tail, continue_src = str(tail), str(tail)
        continue_people = [str(person)]
        def set_stage(self, *_): pass
        def is_cancelled(self): return False
        def remember(self, *a): pass
        def memory_text(self): return ""
    ctx, state = Ctx(), {}
    out = H._handle_generate_video(ctx, state, {"description": "a cat walks in and plays with the dog"})
    assert "[NOT MADE YET]" in str(out) and state.get("video_plan")
    assert ctx.continue_tail == str(tail) and ctx.continue_people == [str(person)]


def test_the_newcomer_walks_into_the_existing_frame():
    # Live 2026-10-10: the camera whipped from the selfie down to the cat, and the cat hit the dog.
    c = V.new_people_clause(2, 1)
    assert "no pan" in c and "steps into the frame" in c and "nobody hits" in c
