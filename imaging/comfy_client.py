"""ComfyUI transport: submit a prompt graph, watch it, collect the result.

Extracted from image.py, where it had grown up alongside 7,800 lines of image
EDITING logic despite being neither about images nor about editing. Three
callers need it — image.py, ideogram.py and video.py — and the latter two were
reduced to function-local `from image import _submit_and_poll` imports placed
there purely to break an import cycle.

What lives here is the whole conversation with ComfyUI and nothing else:

  submit          POST /prompt, with a bounded timeout because that endpoint
                  only enqueues (see _COMFY_SUBMIT_TIMEOUT).
  watch           _ComfyProgress, a websocket subscriber that reports sampling
                  steps live, plus _progress_scope, a thread-local ambient hook
                  so multi-stage pipelines need not thread a callback through
                  every signature.
  serialise       _gpu_slot. The GPU is the one resource that genuinely cannot
                  be shared: two renders at once means VRAM pressure and both
                  finish slower. Reentrant per thread so a pipeline that submits
                  several graphs inside one logical edit cannot deadlock itself.
  collect         _poll_history / _submit_and_collect, which decide when a job
                  is done, lost, or merely slow, and validate the file before
                  anyone is handed a path to it.

Callers supply `validate` when "a usable result" is not an image — video.py
passes its own, because ComfyUI reports a saved mp4 under the same history
"images" key that SaveImage uses.
"""
import json
import logging
import os
import config as _cfg_env   # env_int/env_float: a bad .env value falls back, never crashes the import
import sys
import threading
import time
import uuid
from typing import Optional

import requests

from config import COMFY_URL, OUTPUT_DIR, OUTPUT_DIR_COMFY, COMFY_JOB_TIMEOUT
from utils import throttle_external_calls

logger = logging.getLogger("assistant.comfy")

# Bounded (connect, read) timeout for the ComfyUI /prompt POST. That endpoint only
# ENQUEUES the graph and returns a prompt_id immediately — it never waits for the
# render — so a large read timeout serves no purpose and turns a ComfyUI that accepts
# the socket but stalls before responding into a ~32-minute, cancel-proof worker hang
# (the render itself is bounded separately by the poll loop's wall-clock `timeout`).
_COMFY_SUBMIT_TIMEOUT = (10, 120)

# A ComfyUI that has WEDGED — accepts the TCP connection but never answers — is a
# different failure from one that is simply absent, and it is the expensive one:
# every call pays its full read timeout instead of failing on connect. A draw makes
# several calls in a row, so a wedged server burned ~5 minutes and then surfaced as
# "the request took too long", blaming slowness for what was a dead engine.
# One cheap probe up front turns that into a couple of seconds. Cached briefly so a
# burst of calls in one job does not re-probe each time.
_HEALTH_TIMEOUT = (3, 5)
_HEALTH_TTL = 15.0
_health = {"ok": None, "at": 0.0}


def server_healthy(force: bool = False) -> bool:
    """True if ComfyUI answers /system_stats within a few seconds.

    Deliberately NOT a bare connect check: a wedged server passes that and then
    stalls on read, which is the exact case this exists to catch.
    """
    now = time.time()
    if not force and _health["ok"] is not None and now - _health["at"] < _HEALTH_TTL:
        return _health["ok"]
    try:
        ok = requests.get(f"{COMFY_URL}/system_stats",
                          timeout=_HEALTH_TIMEOUT).status_code == 200
    except Exception as exc:
        logger.warning("ComfyUI health probe failed: %s", exc)
        ok = False
    _health["ok"], _health["at"] = ok, now
    return ok


# Why the last submit on THIS thread came back empty, in words a user can be
# told. Callers used to say only "the render produced no file", and the chat
# model then invented a cause ("a conflict in the animation algorithms",
# live 2026-10-03) while the real one sat in the log.
_FAILURE = threading.local()


def _fail(reason: str) -> None:
    _FAILURE.reason = str(reason)[:600]


def last_failure() -> str:
    """The reason the last ComfyUI job on this thread failed ("" if none)."""
    return getattr(_FAILURE, "reason", "")


def _execution_error_text(msg) -> str:
    """One line from a history 'execution_error' payload (a dict, or text)."""
    if isinstance(msg, dict):
        node = msg.get("node_type") or msg.get("node_id") or "?"
        kind = msg.get("exception_type") or "error"
        text = (msg.get("exception_message") or "").strip().splitlines()
        return f"{node}: {kind}: {text[0] if text else ''}".strip()
    return str(msg)


def _format_comfy_error(resp) -> str:
    """Turn a ComfyUI /prompt error response into a readable one-line reason.

    ComfyUI returns per-node validation errors (e.g. a model not installed); the
    bare status code is useless without them.
    """
    try:
        data = resp.json()
    except Exception:
        return (resp.text or "")[:500]
    node_errors = data.get("node_errors") or {}
    if node_errors:
        msgs = []
        for nid, ne in node_errors.items():
            ct = ne.get("class_type", "?")
            for e in ne.get("errors", []):
                msgs.append(f"node {nid} ({ct}): {e.get('details') or e.get('message')}")
        return "; ".join(msgs) or str(data)[:500]
    err = data.get("error") or {}
    # ComfyUI usually returns error as {"type","message"}, but some paths return a bare
    # string (or other shape). Don't let the error FORMATTER itself raise — that would
    # bury the real reason (e.g. "model not installed") under a generic exception log.
    if isinstance(err, dict):
        return err.get("message") or str(data)[:500]
    return str(err)[:500] or str(data)[:500]


def _upload_image_to_comfy(image_path: str, comfy_url: str) -> Optional[str]:
    """Upload an image file to ComfyUI's /upload/image endpoint.
    Returns the server-side filename on success, None on failure.
    """
    try:
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        with open(image_path, "rb") as f:
            resp = requests.post(
                f"{comfy_url}/upload/image",
                files={"image": (os.path.basename(image_path), f, mime)},
                data={"type": "input", "overwrite": "true"},
                timeout=30,
            )
        if resp.status_code == 200:
            data = resp.json()
            name = data.get("name") or os.path.basename(image_path)
            logger.info("Uploaded previous image to ComfyUI: %s", name)
            return name
        logger.error("ComfyUI upload HTTP %d", resp.status_code)
        return None
    except Exception as exc:
        logger.error("ComfyUI upload error: %s", exc)
        return None


def server_busy() -> bool:
    """True if ComfyUI has anything running or queued right now.

    Used to tell "our task is stuck" apart from "our task is waiting its turn
    behind somebody else's render". A Music3 track is ~70 minutes of GPU; an
    image request queued behind one is not hung, it just has not started.
    Conservative on error: an unreachable server is reported NOT busy, so a
    real outage still lets the caller's deadline fire.

    A TIMEOUT is not an outage, though -- it is the opposite signal. Live,
    2026-09-20: a 41-minute H3 video render (heavy VRAM offload, near-100%
    GPU) had this /queue poll time out ONCE, deep into an otherwise-healthy
    render that finished 40 seconds later; the caller's extend-while-busy
    loop read the exception as "not busy" and abandoned the task 9 seconds
    before the real video was ready -- the user got "cancelled, try again"
    for a render that had already succeeded. A server too saturated by GPU
    work to answer a lightweight status request in time IS busy; only a
    genuine connection failure (refused/DNS/reset) means it might be down.
    """
    try:
        q = requests.get(f"{COMFY_URL}/queue", timeout=(3, 5)).json()
        return bool(q.get("queue_running") or q.get("queue_pending"))
    except requests.exceptions.Timeout:
        return True
    except Exception:
        return False


def _valid_image_file(path: str) -> bool:
    """ComfyUI's history can name a file that is missing, 0-byte, or truncated
    (disk full / AV interference on its side). Never hand such a path to the
    GUI or the edit chain — it would be displayed broken and edited blindly."""
    try:
        # Tiny floor only — legitimate small PNGs (flat colors) compress well
        # under 1KB; truncated/garbage files are caught by PIL's verify below.
        if not os.path.isfile(path) or os.path.getsize(path) < 100:
            return False
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def _job_still_known(prompt_id: str) -> Optional[bool]:
    """Ask ComfyUI whether ``prompt_id`` is still queued or running.

    Returns True (present), False (definitely absent from both queues), or None
    when the queue could not be read (fail open — never abort on a flaky read).
    A job that is in neither the queue nor /history has been dropped (server
    restarted mid-job, queue cleared) and will never finish; polling it for the
    full timeout is the "stuck on Drawing a picture" hang.
    """
    try:
        q = requests.get(f"{COMFY_URL}/queue", timeout=10).json()
        for key in ("queue_running", "queue_pending"):
            for item in q.get(key) or []:
                # queue item shape: [number, prompt_id, prompt, extra, outputs]
                if isinstance(item, (list, tuple)) and len(item) > 1 and item[1] == prompt_id:
                    return True
        return False
    except Exception:
        return None


def _cancel_job(prompt_id: str) -> None:
    """Stop OUR job on the server, not just our wait for it. A cancelled poll
    used to leave the render running: the chat model came back into a card
    with 360 MiB free and every next turn hung 5 min (live 2026-09-28).
    Interrupt only when ours is the one executing -- never another user's."""
    try:
        q = requests.get(f"{COMFY_URL}/queue", timeout=10).json()
        running = [i[1] for i in q.get("queue_running") or [] if isinstance(i, (list, tuple)) and len(i) > 1]
        if prompt_id in running:
            requests.post(f"{COMFY_URL}/interrupt", json={"prompt_id": prompt_id}, timeout=10)
        else:
            requests.post(f"{COMFY_URL}/queue", json={"delete": [prompt_id]}, timeout=10)
        for _ in range(60):                 # the sampler stops between steps
            if not _job_still_known(prompt_id):
                return
            time.sleep(0.5)
    except Exception as exc:
        logger.info("ComfyUI cancel of %s failed (%s)", prompt_id, exc)


class _ComfyProgress:
    """Live sampling progress (step k of n) from ComfyUI's websocket.

    /history only shows a job once it is FINISHED, so the step counter cannot come
    from the poll loop — ComfyUI pushes it over ws://<host>/ws as
    {"type":"progress","data":{"value":k,"max":n,"prompt_id":...}}. This runs that
    socket on its own daemon thread and hands each update to `callback(k, n)`.

    Entirely best-effort: if websocket-client is missing or the socket drops, the
    render still runs and polls exactly as before — only the counter goes quiet.
    """

    def __init__(self, client_id: str, callback, prompt_id_holder):
        self.client_id = client_id
        self.callback = callback
        self.holder = prompt_id_holder      # {"id": ...}, filled in after /prompt
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        try:
            import websocket            # noqa: F401  (presence check)
        except Exception:
            logger.info("websocket-client not installed — no live step counter")
            return self
        self._thread = threading.Thread(target=self._run, name="comfy-progress",
                                        daemon=True)
        self._thread.start()
        return self

    def _run(self):
        import websocket
        url = COMFY_URL.replace("https://", "wss://").replace("http://", "ws://")
        try:
            ws = websocket.create_connection(f"{url}/ws?clientId={self.client_id}",
                                             timeout=10)
        except Exception as exc:
            logger.info("ComfyUI progress socket unavailable: %s", exc)
            return
        try:
            while not self._stop.is_set():
                try:
                    raw = ws.recv()
                except Exception:
                    return                              # socket closed or timed out
                if not raw or isinstance(raw, (bytes, bytearray)):
                    continue                            # binary frames are previews
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                data = msg.get("data") or {}
                want = self.holder.get("id")
                if want and data.get("prompt_id") not in (None, want):
                    continue                            # another client's job
                mtype = msg.get("type")
                if mtype == "progress":
                    # Legacy flat message. Kept for older ComfyUI builds.
                    try:
                        self.callback(int(data.get("value", 0)), int(data.get("max", 0)))
                    except Exception:
                        logger.exception("progress callback failed")
                elif mtype == "progress_state":
                    # What ComfyUI actually sends now (comfy_execution/progress.py
                    # only ever emits "progress_state"). It is a COMBINED message —
                    # {"nodes": {node_id: {"value": k, "max": n, "state": ...}}} —
                    # so the flat branch above had been matching nothing and the
                    # live step counter was silently dead for every render, images
                    # included. Report the node that is actually sampling: prefer a
                    # running one, and ignore max<=1 nodes (loaders tick 0/1 and
                    # would otherwise yank the bar back to zero between steps).
                    best = None
                    _nodes = data.get("nodes")
                    if not isinstance(_nodes, dict):
                        continue        # malformed frame; never kill the socket
                    for _nid, st in _nodes.items():
                        if not isinstance(st, dict):
                            continue
                        try:
                            mx = int(st.get("max") or 0)
                            val = int(st.get("value") or 0)
                        except (TypeError, ValueError):
                            continue
                        if mx <= 1:
                            continue
                        running = str(st.get("state") or "").lower() == "running"
                        # A running node wins; otherwise keep the furthest-along one.
                        rank = (1 if running else 0, val)
                        if best is None or rank > best[0]:
                            best = (rank, val, mx)
                    if best is not None:
                        try:
                            self.callback(best[1], best[2])
                        except Exception:
                            logger.exception("progress callback failed")
                elif mtype == "executing" and data.get("node") is None:
                    return                              # this prompt is done
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def stop(self):
        self._stop.set()


# Ambient progress hook. A multi-stage edit (inpaint_region_with_comfy) fans out to
# several _submit_and_poll calls through helper functions that do not carry an
# on_progress parameter. Rather than thread the callback through every signature,
# the outer call installs one here and every _submit_and_poll on THIS thread picks
# it up. Thread-local so concurrent workers never cross wires.
_PROGRESS_HOOK = threading.local()


class _progress_scope:
    """Install a fallback on_progress for every _submit_and_poll on this thread."""

    def __init__(self, callback):
        self.callback = callback

    def __enter__(self):
        self._prev = getattr(_PROGRESS_HOOK, "cb", None)
        _PROGRESS_HOOK.cb = self.callback
        return self

    def __exit__(self, *exc):
        _PROGRESS_HOOK.cb = self._prev
        return False


# ── the one genuinely exclusive resource ──────────────────────────────────────
# The Telegram bot now runs several tasks at once so an LLM turn can proceed
# while somebody else's picture renders. The GPU cannot be shared that way: two
# ComfyUI jobs at once means VRAM pressure and both finishing slower than if they
# had queued. This semaphore is what makes the concurrency safe — everything else
# (LLM calls, web crawling, transcription, file work) overlaps freely, and only
# the render serialises.
COMFY_MAX_CONCURRENT = max(1, _cfg_env.env_int("COMFY_MAX_CONCURRENT", 1))
_GPU_SEM = threading.Semaphore(COMFY_MAX_CONCURRENT)


# ── giving the card to the render ────────────────────────────────────────────
# A 24 GB card cannot hold a 20 GB chat model AND a render at the same time, and
# what happens instead of an error is worse than one: ComfyUI stages the weights
# it cannot fit and streams them over the bus, so the job merely gets slow.
# Measured on one 30-second song, same seed, same preset (bench/music_vram_ab.py):
#
#     card to ourselves .......  43.5 s   (44.6 AR it/s)
#     chat model resident .....  69.3 s   (17.3 AR it/s)   1.59x
#     after an image render ...  273.6 s  ( 3.0 AR it/s)   6.3x
#
# So the card is claimed for the duration of a render BATCH: evicted once when
# the first job starts, and the chat model is only brought back when the queue
# behind it has drained. Restoring per job would pay the reload on every picture
# of a five-picture batch.
#
# Opt-in per call site, NOT global: several edit pipelines call a vision model
# BETWEEN their ComfyUI submits, and evicting under those would break the very
# turn that asked for the edit. Only the paths that need nothing but the GPU
# once they start pass exclusive=True.
COMFY_EVICT_LLM = os.getenv("COMFY_EVICT_LLM", "1") not in ("0", "false", "False", "")
# How long a claim waits for open LLM streams to finish before unloading the
# model anyway (a reply mid-sentence is worth a short wait; a stuck stream is not).
LLM_INFLIGHT_WAIT_S = _cfg_env.env_int("LLM_INFLIGHT_WAIT_S", 120)

_EXCL_LOCK = threading.RLock()
_excl_users = 0            # jobs holding OR queued for an exclusive slot
_excl_model = ""           # the chat model we unloaded ("" = we unloaded nothing)
_excl_label = ""           # what the card is busy with, for the waiting message
_excl_free = threading.Event()
_excl_free.set()
EXCL_NOTICE = None         # optional callback(str) -- surfaces show it to the user


def _notice(msg):
    logger.info("%s", msg)
    cb = EXCL_NOTICE
    if cb is None:
        return
    try:
        cb(msg)
    except Exception:
        logger.exception("exclusive-card notice callback raised")


def card_is_exclusive() -> str:
    """What the card is exclusively rendering, or "" when it is shared.

    A chat turn asks this to explain the wait instead of quietly triggering a
    20 GB reload that would fight the render it is waiting for.
    """
    with _EXCL_LOCK:
        return _excl_label if _excl_model else ""


def wait_for_card(timeout: float = 600.0) -> bool:
    """Block until the chat model is back. True if the card is free to share."""
    return _excl_free.wait(timeout)


def _claim_card(label: str, min_free_mb: int = None) -> None:
    global _excl_users, _excl_model, _excl_label
    with _EXCL_LOCK:
        _excl_users += 1
        if _excl_users > 1:                 # a batch is already running
            _excl_label = label or _excl_label
            return
        _excl_label = label
        if not COMFY_EVICT_LLM:
            return
        # A test run must never unload the operator's live model. Suites that
        # drive the render path with a stubbed ComfyUI would otherwise reach
        # `lms unload --all` for real -- which is how this guard came to exist.
        # A suite testing THIS mechanism stubs lora_training and is unaffected.
        _lt = sys.modules.get("lora_training")
        _stubbed = _lt is not None and getattr(_lt, "__file__", None) is None
        if os.getenv("F5_TEST_RUN") and not _stubbed:
            return
        try:
            import lora_training as _LT
            model = _LT.loaded_llm_id()
            if not model:
                return
            # Let the streams that are open finish first (the second user's
            # planner, a chat reply mid-sentence): pulling the model out from
            # under them loses their turn and triggers a reload into the
            # render's card. New calls already wait on card_is_exclusive.
            _excl_free.clear()
            _excl_model = model             # from here on, new LLM calls wait
            try:
                import llm as _llm
                if _llm.inflight() and not _llm.wait_no_inflight(LLM_INFLIGHT_WAIT_S):
                    logger.warning("an LLM stream is still open after %ds - "
                                   "freeing the card anyway", LLM_INFLIGHT_WAIT_S)
            except Exception:
                logger.exception("could not wait for open LLM streams")
            _LT.free_gpu(log=logger.info, min_free_mb=min_free_mb)
            _notice("card freed for %s (chat model %s unloaded)" % (label, model))
        except Exception:
            # Never let VRAM housekeeping cost somebody their render.
            logger.exception("could not free the card - rendering anyway")
            _excl_model = ""
            _excl_free.set()


def _free_comfy_models() -> None:
    """Ask ComfyUI to drop its resident models. Best effort, never raises."""
    try:
        import urllib.request as _ur
        req = _ur.Request(COMFY_URL.rstrip("/") + "/free",
                          data=json.dumps({"unload_models": True,
                                           "free_memory": True}).encode(),
                          headers={"Content-Type": "application/json"})
        _ur.urlopen(req, timeout=30).read()
    except Exception as exc:
        logger.info("ComfyUI /free did not answer (%s) - continuing", exc)


# Room the chat model needs to come back with the headroom it had: the 26B at
# its working context plus the vision tokens of one picture. Below FREE_GPU_MIN_MB
# (16 GB) is "still draining"; this is "enough to serve".
RELOAD_LLM_MIN_FREE_MB = _cfg_env.env_int("RELOAD_LLM_MIN_FREE_MB", 19000)
# Extra wait, after a second /free, when the first FREE_GPU_WAIT_S was not
# enough. Loading the 18 GB model into a card ComfyUI has not vacated makes
# the driver spill it into shared system RAM: the load crawls (minutes at
# 90 %) and every answer after it is slow -- worse than waiting here.
RELOAD_VRAM_EXTRA_WAIT_S = _cfg_env.env_float("RELOAD_VRAM_EXTRA_WAIT_S", 90)


def _release_card() -> None:
    global _excl_users, _excl_model, _excl_label
    with _EXCL_LOCK:
        _excl_users = max(0, _excl_users - 1)
        if _excl_users:                     # more of the batch is still queued
            return
        model, _excl_model, _excl_label = _excl_model, "", ""
    if not model:
        _excl_free.set()
        return
    try:
        import lora_training as _LT
        _notice("render queue empty - bringing %s back" % model)
        # ComfyUI FIRST. It still holds the render's weights, and a chat model
        # loaded into what they leave behind comes up short of the headroom it
        # had before the render -- on 2026-09-11 it loaded, then crashed on
        # the first vision call (a 1728x2304 inspect) and every turn after
        # that was "No models loaded". Same order the claim used, mirrored.
        _free_comfy_models()
        # And WAIT for the driver to have that memory back: /free returns
        # before the tensors are gone, the same way `lms unload` does on the
        # way in. The model still needs the room it had before the render;
        # loaded a few hundred MB short it comes up, then dies on the first
        # picture (live, 2026-09-12, journey 3: crash on a vision call two
        # minutes after the give-back, three turns lost).
        free = _LT.wait_vram_free(RELOAD_LLM_MIN_FREE_MB, timeout=_LT.FREE_GPU_WAIT_S)
        if free is not None and free < RELOAD_LLM_MIN_FREE_MB and RELOAD_VRAM_EXTRA_WAIT_S > 0:
            # ComfyUI can take its time letting go (a second queue item, a
            # slow /free): ask again and give it longer before squeezing in.
            _notice("waiting for ComfyUI to free the card (%d MiB free, %d needed)"
                    % (free, RELOAD_LLM_MIN_FREE_MB))
            _free_comfy_models()
            free = _LT.wait_vram_free(RELOAD_LLM_MIN_FREE_MB, timeout=RELOAD_VRAM_EXTRA_WAIT_S)
            if free is not None and free < RELOAD_LLM_MIN_FREE_MB:
                logger.warning("reloading %s with only %d MiB free (%d wanted): it may spill "
                               "into shared memory and load and answer slowly", model, free,
                               RELOAD_LLM_MIN_FREE_MB)
        _LT.reload_llm(model, log=logger.info)
    except Exception:
        logger.exception("could not bring the chat model back")
    finally:
        _excl_free.set()


class card_session:
    """Hold the card across a WHOLE sequence of renders, not one submit.

    The collage repair chain redraws the same layout up to five times, deciding
    purely on pixels, with no model call in between. Claiming per submit made it
    unload and reload a 20 GB model between every attempt -- about 18 s of pure
    waste each time, for a decision that never needed the model at all.

    NOT for a loop that consults a model between renders (the layout critic
    reads the picture back and edits the boxes): inside a session the chat model
    is gone, and llm.send_to_lm_studio would wait for a session that is waiting
    for it. The thread-local flag below is what keeps that a no-op rather than a
    deadlock, but a critic call in here would still run with nothing loaded.
    """
    _local = threading.local()

    def __init__(self, label: str = ""):
        self.label = label

    def __enter__(self):
        _claim_card(self.label)
        card_session._local.held = getattr(card_session._local, "held", 0) + 1
        return self

    def __exit__(self, *exc):
        card_session._local.held = getattr(card_session._local, "held", 1) - 1
        _release_card()
        return False


def this_thread_holds_card() -> bool:
    """True when THIS thread is the one the card was given to.

    Such a thread must never wait for the card: it would be waiting for itself.
    """
    return bool(getattr(card_session._local, "held", 0)
                or getattr(_gpu_slot._local, "depth", 0))


class _gpu_slot:
    """Hold a render slot. Reentrant per thread: a pipeline that submits several
    graphs inside one logical edit must not deadlock against itself.

    `exclusive` additionally gives the card to the render by evicting the chat
    model -- see COMFY_EVICT_LLM above for why that is opt-in.
    """
    _local = threading.local()

    def __init__(self, exclusive: bool = False, label: str = "", min_free_mb: int = None):
        self.exclusive = bool(exclusive)
        self.label = label
        self.min_free_mb = min_free_mb

    def __enter__(self):
        depth = getattr(self._local, "depth", 0)
        self._acquired = (depth == 0)
        # Claimed BEFORE the semaphore, so a job waiting its turn keeps the card
        # claimed and the model is not reloaded in the gap between two renders.
        self._claimed = self.exclusive and self._acquired
        if self._claimed:
            _claim_card(self.label, min_free_mb=self.min_free_mb)
        if self._acquired:
            _GPU_SEM.acquire()
        self._local.depth = depth + 1
        return self

    def __exit__(self, *exc):
        self._local.depth = getattr(self._local, "depth", 1) - 1
        if self._acquired:
            _GPU_SEM.release()
        if self._claimed:
            _release_card()
        return False


# Set by _submit_and_poll each time it actually declines a job because a
# foreign whole-card job holds runtime/gpu.lock. Read at the tool boundary to
# explain the failure. A TIMESTAMP rather than a flag: "is the card busy right
# now" is the wrong question there -- Ideogram can refuse a prompt on content
# grounds during a training run, and blaming the trainer for that would be a
# confident wrong answer.
LAST_GPU_REFUSAL: dict = {"label": None, "at": 0.0}


def recent_gpu_refusal(within: float = 180.0):
    """The label of a job we declined for the GPU lock in the last `within`
    seconds, or None."""
    if not LAST_GPU_REFUSAL.get("label"):
        return None
    if time.time() - float(LAST_GPU_REFUSAL.get("at") or 0) > within:
        return None
    return LAST_GPU_REFUSAL["label"]


try:                       # captured at import, before any suite redirects it
    import gpu_lock as _gpu_lock_mod
    _REAL_LOCK_PATH = _gpu_lock_mod.LOCK_PATH
except Exception:
    _REAL_LOCK_PATH = None


def gpu_holder():
    """The label of a FOREIGN whole-card job holding the GPU, or None.

    gpu_lock has existed since a second trainer was launched on top of a live
    one, but until now only the bench scripts consulted it -- the app and the
    bot drew straight through a training run. On a 24 GB card that is not a
    queue, it is a fight: the training step time nearly doubles and the render
    tends to die on an OOM anyway. Reading it costs a file stat.

    Returns None (card free, or held by THIS process) rather than raising, so a
    caller can treat "cannot tell" as "go ahead".
    """
    try:
        import config as _cfg
        if not getattr(_cfg, "COMFY_RESPECT_GPU_LOCK", True):
            return None
        import gpu_lock
        # A test run must not read the operator's live card. Suites that stub
        # ComfyUI and drive _submit_and_poll would otherwise pass or fail
        # depending on whether a training run happened to be going -- which is
        # exactly what happened the first time this gate landed. A suite that
        # REDIRECTS the lock path is testing this mechanism on purpose, so it
        # is let through; the same convention as lmstudio._refuse_under_tests.
        if os.getenv("F5_TEST_RUN") and gpu_lock.LOCK_PATH == _REAL_LOCK_PATH:
            return None
        held = gpu_lock.owner()
    except Exception:
        return None
    return held[1] if held else None


def _submit_and_poll(ctx, workflow: dict, timeout: int = 1900, label: str = "",
                     on_progress=None, validate=None, job_timeout=None,
                     exclusive: bool = False, min_free_mb: int = None) -> Optional[str]:
    """POST a ComfyUI prompt graph, poll /history until an image is saved.

    `on_progress(step, total)` — optional; called live during sampling (see
    `_ComfyProgress`). Never let it raise: it usually crosses into a GUI thread.

    `validate(path) -> bool` — what counts as a usable result file. Defaults to
    `_valid_image_file`. video.py passes its own: a SaveVideo job reports its mp4
    under the SAME "images" history key as SaveImage (ComfyUI's PreviewVideo emits
    {"images": [...], "animated": true}), so without this the path would be found
    and then thrown away for failing an IMAGE integrity check.

    `exclusive` — give the card to this job: the chat model is unloaded first and
    brought back when the render queue drains. Only for paths that need nothing
    but the GPU once they start; an edit pipeline that consults a vision model
    between submits must NOT set it.

    `job_timeout` — the hard ceiling to clamp against, default config.COMFY_JOB_TIMEOUT.
    A video clip legitimately runs far longer than any still render, so video.py
    raises it to config.VIDEO_JOB_TIMEOUT rather than being silently cut off.

    `min_free_mb` — how much free VRAM to insist on before this job's first
    submit, overriding lora_training.FREE_GPU_MIN_MB (16 GB, calibrated only
    for confirming the chat model has drained). Live, 2026-09-19: H3 video
    renders that started with 16-17 GB free sometimes took 35-49 minutes
    instead of their usual 8-12, consistent with ComfyUI having to swap the
    unet/CLIP/VAEs in and out of VRAM mid-render rather than holding them all
    resident. video.py passes a higher floor for exactly this reason.

    Returns the saved file path, or None on HTTP error / job error / timeout.
    """
    timeout = min(timeout, job_timeout if job_timeout is not None else COMFY_JOB_TIMEOUT)
    _fail("")
    if ctx is not None and getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
        # Stop pressed while the prompt was being planned: claiming the card
        # would unload the chat model for a job nobody wants (live 2026-09-28).
        logger.info("ComfyUI submit skipped: cancelled%s", f" ({label})" if label else "")
        return None
    if not server_healthy():
        logger.error("ComfyUI is not answering — abandoning submit%s",
                     f" ({label})" if label else "")
        _fail("ComfyUI is not running or not answering at " + COMFY_URL)
        return None
    busy = gpu_holder()
    if busy:
        # Do not queue behind it and do not fight it: a whole-card job runs for
        # hours, and every caller here has a timeout measured in minutes.
        logger.error("GPU held by %r — refusing this ComfyUI job%s", busy,
                     f" ({label})" if label else "")
        LAST_GPU_REFUSAL.update({"label": busy, "at": time.time()})
        _fail(f"the GPU is busy with {busy}")
        return None
    with _gpu_slot(exclusive=exclusive, label=label, min_free_mb=min_free_mb):
        return _submit_and_poll_locked(ctx, workflow, timeout, label, on_progress, validate)


# Files the graphs name that are not published anywhere (local quantizations),
# with the public files the setup script downloads in their place. A graph
# whose file is missing on the server is sent with the stand-in, so a fresh
# install renders instead of failing "value not in list".
MODEL_FALLBACKS = {
    ("UNETLoader", "unet_name"): {
        "ig4-int8mixedrow_simple.safetensors": "ideogram4_fp8_scaled.safetensors",
        "ig4_uncond-int8mixedrow_simple.safetensors": "ideogram4_unconditional_fp8_scaled.safetensors",
    },
}
_AVAILABLE: dict = {}          # (class, input) -> (time, set of names)


def _available_models(node_class: str, field: str) -> Optional[set]:
    hit = _AVAILABLE.get((node_class, field))
    if hit and time.time() - hit[0] < 60:
        return hit[1]
    try:
        r = requests.get(f"{COMFY_URL}/object_info/{node_class}", timeout=10)
        spec = r.json()[node_class]["input"]["required"][field]
        names = set(spec[0] if isinstance(spec[0], list) else spec[1].get("options", []))
    except Exception:
        return None
    _AVAILABLE[(node_class, field)] = (time.time(), names)
    return names


def _apply_model_fallbacks(workflow: dict) -> dict:
    """Swap a model file the server lacks for its published stand-in."""
    for node in (workflow or {}).values():
        if not isinstance(node, dict):
            continue
        for (cls, field), subs in MODEL_FALLBACKS.items():
            want = (node.get("inputs") or {}).get(field)
            if node.get("class_type") != cls or want not in subs:
                continue
            have = _available_models(cls, field)
            if have is not None and want not in have and subs[want] in have:
                logger.info("ComfyUI has no %s -- using %s", want, subs[want])
                node["inputs"][field] = subs[want]
    return workflow


def _submit_and_poll_locked(ctx, workflow: dict, timeout: int, label: str,
                            on_progress, validate=None) -> Optional[str]:
    if on_progress is None:
        on_progress = getattr(_PROGRESS_HOOK, "cb", None)  # ambient hook (see _progress_scope)
    # Start the progress socket BEFORE submitting: subscribing with the same client_id
    # that /prompt is tagged with means we do not miss the first step. The prompt_id it
    # filters on is filled into `holder` the moment /prompt answers.
    client_id = uuid.uuid4().hex
    holder = {"id": None}
    progress = None
    if on_progress is not None:
        progress = _ComfyProgress(client_id, on_progress, holder).start()
    try:
        throttle_external_calls(ctx)
        _apply_model_fallbacks(workflow)
        resp = requests.post(f"{COMFY_URL}/prompt",
                             json={"prompt": workflow, "client_id": client_id},
                             timeout=_COMFY_SUBMIT_TIMEOUT)
        if resp.status_code != 200:
            _why = _format_comfy_error(resp)
            logger.error("ComfyUI HTTP %d: %s", resp.status_code, _why)
            _fail(f"ComfyUI rejected the job: {_why}")
            if progress is not None:
                progress.stop()
            return None

        resp_json = resp.json()
        prompt_id = resp_json.get("prompt_id")
        if not prompt_id:
            logger.error("ComfyUI response missing prompt_id: %s", resp_json)
            if progress is not None:
                progress.stop()
            return None
        holder["id"] = prompt_id
    except Exception as exc:
        logger.error("ComfyUI request failed: %s", exc)
        if progress is not None:
            progress.stop()
        return None

    logger.info("ComfyUI job started (ID: %s%s)", prompt_id, f", {label}" if label else "")

    try:
        return _poll_history(ctx, prompt_id, timeout, label, validate)
    finally:
        if progress is not None:
            progress.stop()


def _poll_history(ctx, prompt_id, timeout, label, validate=None):
    start = time.time()
    delay = 1.0
    # If ComfyUI dies mid-job the poll keeps failing — by connection-refusal if the
    # process is gone, by read timeout if it is wedged; without this a dead server
    # would be polled for the whole `timeout` (up to 31 min). Tolerate a brief
    # restart, but bail once it has been unreachable continuously for 60s.
    conn_down_since = None
    CONN_FAIL_GRACE = 60.0
    # Vanished-job detection: a job absent from BOTH /queue and /history (server
    # restarted mid-render, queue cleared) will never complete — stop polling after
    # a few consecutive confirmations instead of waiting out the full timeout.
    queue_absent = 0
    while time.time() - start < timeout:
        if ctx is not None and getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
            logger.info("ComfyUI poll cancelled by user%s", f" ({label})" if label else "")
            _cancel_job(prompt_id)
            return None
        try:
            hist_resp = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)
            conn_down_since = None  # reachable again — reset the outage timer
            if hist_resp.status_code == 200:
                history = hist_resp.json()
                if prompt_id not in history and time.time() - start > 10:
                    known = _job_still_known(prompt_id)
                    if known is False:
                        queue_absent += 1
                        if queue_absent >= 3:
                            logger.error("ComfyUI job %s vanished (not in queue or history "
                                         "for 3 consecutive checks) — aborting poll%s",
                                         prompt_id, f" ({label})" if label else "")
                            _fail("the job disappeared from ComfyUI (it restarted or crashed "
                                  "mid-render, often out of memory)")
                            return None
                    elif known:
                        queue_absent = 0
                if prompt_id in history:
                    job = history[prompt_id]
                    outputs = job.get("outputs") or {}
                    job_status = job.get("status", {})

                    last_path = None
                    for node_out in outputs.values():
                        # SaveImage/PreviewImage/SaveVideo report under "images"; audio
                        # nodes (SaveAudio, SaveAudioAdvanced) report under "audio" —
                        # same {filename, subfolder} shape, different key. Without this,
                        # a genuinely successful audio job reads as "no output" because
                        # the loop only ever looked for "images".
                        items = node_out.get("images") or node_out.get("audio")
                        if items:
                            item = items[0]
                            filename = item["filename"]
                            subfolder = item.get("subfolder", "")
                            path = (os.path.join(OUTPUT_DIR_COMFY, subfolder, filename)
                                    if subfolder else os.path.join(OUTPUT_DIR_COMFY, filename))
                            # Skip ComfyUI temp/preview files (PreviewImage nodes write
                            # "ComfyUI_temp_*" files); only SaveImage writes the real output.
                            # Keep updating so the LAST non-temp file wins (graph order).
                            # A LoadVideo preview echoes the INPUT file (type "input").
                            if not filename.lower().startswith("comfyui_temp") \
                                    and item.get("type") != "input":
                                last_path = path
                    if last_path:
                        _ok = validate or _valid_image_file
                        if _ok(last_path):
                            logger.info("Output saved: %s", last_path)
                            return last_path
                        logger.error("ComfyUI reported an output but the file is "
                                     "missing or corrupt: %s", last_path)
                        _fail(f"ComfyUI wrote a missing or broken file: {last_path}")
                        return None

                    # Terminal with no usable output — exit immediately. ComfyUI marks
                    # a SUCCESSFUL-but-empty job with completed=True, but an ERRORED job
                    # with completed=False and status_str=="error" (and an execution_error
                    # message). Both are terminal: without the error check a failed node
                    # would be polled for the full `timeout` (up to 31 min hang).
                    status_str = job_status.get("status_str")
                    errored = status_str == "error" or any(
                        isinstance(m, (list, tuple)) and m and m[0] == "execution_error"
                        for m in job_status.get("messages", []))
                    if job_status.get("completed") or errored:
                        err_parts = [
                            _execution_error_text(m[1]) for m in job_status.get("messages", [])
                            if isinstance(m, (list, tuple)) and len(m) >= 2
                            and m[0] in ("execution_error", "execution_interrupted")
                        ]
                        _why = "; ".join(str(p) for p in err_parts) or "unknown error"
                        logger.error("ComfyUI job %s with no output: %s",
                                     "errored" if errored else "completed", _why)
                        _fail(f"ComfyUI failed during the render: {_why}" if errored
                              else "ComfyUI finished the job without saving a file")
                        return None
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as exc:
            # Timeout as well as ConnectionError: requests' ReadTimeout does NOT
            # subclass ConnectionError, so a WEDGED server (socket accepted, no
            # answer) used to skip this grace entirely and be retried until the
            # full wall-clock `timeout` ran out.
            now = time.time()
            if conn_down_since is None:
                conn_down_since = now
            elif now - conn_down_since > CONN_FAIL_GRACE:
                logger.error("ComfyUI unreachable for >%.0fs (server down?) — aborting poll: %s",
                             CONN_FAIL_GRACE, exc)
                _fail("ComfyUI stopped answering mid-render (crashed or restarted)")
                return None
            logger.warning("ComfyUI connection error (down %.0fs): %s", now - conn_down_since, exc)
        except Exception as exc:
            logger.error("ComfyUI polling error: %s", exc)

        time.sleep(delay)
        delay = min(delay * 1.5, 5.0)

    logger.error("ComfyUI timeout after %d seconds", timeout)
    _fail(f"the render did not finish within {timeout} s")
    return None


def _submit_and_collect(ctx, workflow: dict, timeout: int = 1900,
                        label: str = "") -> Optional[dict]:
    """Like _submit_and_poll, but returns a dict {node_id: saved_path} for EVERY
    SaveImage output of the job (not just the last). Used by multi-output graphs
    (e.g. the hand refiner's detect pass that saves both a depth map and a mask).
    Returns None on HTTP/job error or timeout. ``timeout`` is clamped to
    config.COMFY_JOB_TIMEOUT like _submit_and_poll."""
    timeout = min(timeout, COMFY_JOB_TIMEOUT)
    if not server_healthy():
        logger.error("ComfyUI is not answering — abandoning submit%s",
                     f" ({label})" if label else "")
        return None
    with _gpu_slot():
        return _submit_and_collect_locked(ctx, workflow, timeout, label)


def _submit_and_collect_locked(ctx, workflow: dict, timeout: int,
                               label: str) -> Optional[dict]:
    try:
        throttle_external_calls(ctx)
        _apply_model_fallbacks(workflow)
        resp = requests.post(f"{COMFY_URL}/prompt", json={"prompt": workflow},
                             timeout=_COMFY_SUBMIT_TIMEOUT)
        if resp.status_code != 200:
            logger.error("ComfyUI HTTP %d: %s", resp.status_code, _format_comfy_error(resp))
            return None
        prompt_id = resp.json().get("prompt_id")
        if not prompt_id:
            logger.error("ComfyUI response missing prompt_id")
            return None
    except Exception as exc:
        logger.error("ComfyUI request failed: %s", exc)
        return None

    logger.info("ComfyUI job started (ID: %s%s)", prompt_id, f", {label}" if label else "")
    start, delay = time.time(), 1.0
    while time.time() - start < timeout:
        if ctx is not None and getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
            return None
        try:
            hist = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)
            if hist.status_code == 200 and prompt_id in hist.json():
                job = hist.json()[prompt_id]
                outputs = job.get("outputs") or {}
                collected = {}
                for nid, node_out in outputs.items():
                    for img in (node_out.get("images") or []):
                        fn = img["filename"]
                        if fn.lower().startswith("comfyui_temp"):
                            continue
                        sub = img.get("subfolder", "")
                        p = (os.path.join(OUTPUT_DIR_COMFY, sub, fn) if sub
                             else os.path.join(OUTPUT_DIR_COMFY, fn))
                        if _valid_image_file(p):
                            collected[nid] = p
                jstat = job.get("status", {})
                errored = jstat.get("status_str") == "error"
                if collected and (jstat.get("completed") or not errored):
                    return collected
                if jstat.get("completed") or errored:
                    logger.error("ComfyUI multi-output job ended with no usable images")
                    return None
        except Exception as exc:
            logger.error("ComfyUI collect polling error: %s", exc)
        time.sleep(delay)
        delay = min(delay * 1.5, 5.0)
    logger.error("ComfyUI collect timeout after %d seconds", timeout)
    return None


def adopt_output(path: str, what: str = "file", out_dir=None) -> str:
    """Copy a finished render out of ComfyUI's output tree into ours.

    ComfyUI's output directory is periodically cleaned and is shared with every
    other graph, so anything we are about to hand to a user should not live
    there. Returns the new path, or the original one if the copy was impossible
    (delivering the file in place beats failing a whole job over a copy).

    The collision guard is load-bearing, not defensive padding: ComfyUI derives
    its `00001_` filename counter from what is in ITS output tree, so once that
    tree is cleaned the numbering RESTARTS — and a blind copyfile() would then
    overwrite a render we had already kept for the user. Renders cost minutes of
    GPU; never silently destroy one.

    `what` names the artifact in the failure log ("song", "clip", …).

    `out_dir` is the destination, defaulting to config.OUTPUT_DIR. Callers pass
    THEIR OWN module-level OUTPUT_DIR rather than relying on the default: each
    caller's global is what the suites rebind to a temp directory, and a shared
    helper that read only its own copy would ignore that patch and drop fixture
    files into the user's real output tree.

    This is the single definition. music.py and video.py had each grown their own
    byte-identical copy, and only video's carried the paragraph above — so the
    music copy had the guard with none of the reasoning that explains why it has
    to stay.
    """
    import shutil
    dest_dir = OUTPUT_DIR if out_dir is None else out_dir
    try:
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(path))
        if os.path.abspath(dest) == os.path.abspath(path):
            return path
        if os.path.exists(dest):
            stem, ext = os.path.splitext(dest)
            n = 2
            while os.path.exists(f"{stem}_{n}{ext}"):
                n += 1
            dest = f"{stem}_{n}{ext}"
        shutil.copyfile(path, dest)
        return dest
    except Exception as exc:
        logger.warning("could not copy the %s into %s (%s) — using it in place",
                       what, dest_dir, exc)
    return path
