"""Out-of-process backend for deep research: a small HTTP service plus a client
that implements the same contract (research_api.ResearchClient).

Nothing selects it unless you ask for it:

    set RESEARCH_BACKEND=http
    set RESEARCH_SERVICE_URL=http://127.0.0.1:8791
    venv\\Scripts\\python.exe research_service.py --port 8791

The default remains the in-process backend, so an unconfigured install is
byte-for-byte the product it was before this branch.

WHY THIS ONE IS ALLOWED OUT OF PROCESS AT ALL
----------------------------------------------
Because the analysis (docs/gui_service_boundaries.md) showed both hard parts
survive the hop, and this module is what makes them survive:

  PROGRESS  deep_research's callback is progress(phase: str, stats: dict,
            msg: str) — every argument already JSON-serialisable. So /op/run
            does not return a single JSON body; it STREAMS one NDJSON record
            per tick and the CLIENT re-invokes the caller's local callable. The
            GUI's phase-by-phase progress strip keeps working. A request/response
            route would have collapsed it to "started" and "done", which is a
            visible regression, not an implementation detail.

  CANCEL    ctx.cancel_event is a process-local threading.Event, so the server
            cannot see the GUI's Stop. The service therefore issues a RUN ID as
            the first record of the stream and holds its OWN Event, which a
            _ServerCtx hands to the pipeline via is_cancelled(). The client runs
            a watchdog thread that polls the caller's real ctx and fires
            POST /op/cancel the moment Stop is pressed. Cost: up to
            CANCEL_POLL_S of extra latency plus one round-trip; behaviour:
            identical (deep_research returns its cancelled=True result and the
            stream ends normally). HttpResearchClient.cancellation_mode is
            "run_id", not "shared_event", and says so.

WHAT THE HTTP BACKEND STILL CANNOT DO, AND SAYS SO
---------------------------------------------------
* `result["path"]` is a path on the machine that RAN the research. Same-host
  deployment only. A client on another host gets a path it cannot open; the
  report text itself is in result["report"] and is complete, so a remote-host
  caller must use that and ignore the path.
* Overrides are applied to the SERVICE process's deep_research globals for the
  duration of one run. apply/restore is per-run and in a finally, but it is
  still module-global state: the service therefore serialises runs behind
  _RUN_LOCK rather than pretending concurrent runs with different knobs are
  safe. That is a deliberate throughput limit, not an oversight.

DEPENDENCIES
------------
Server: stdlib only (http.server + json + threading), matching knowledge_service.
Client: `requests`, already a hard dependency of this project.

WIRE PROTOCOL
-------------
    GET  /health           -> {"ok": true, "backend": "http", "ops": [...]}
    POST /op/<op>          -> {"ok": true, "result": <json>}  (non-streaming ops)
                           or {"ok": false, "error": "<Type: message>"}
    POST /op/run           -> NDJSON stream, one JSON object per line:
                                {"type": "run_id",   "run_id": "..."}
                                {"type": "progress", "phase":..,"stats":..,"msg":..}
                                {"type": "result",   "result": {...}}
                              or {"type": "error",   "error": "..."}
    POST /op/cancel        -> {"ok": true, "result": true|false}
    POST /shutdown         -> {"ok": true}
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from http.server import ThreadingHTTPServer
from service_http import JsonHandler
from typing import Any, Callable, Dict, List, Optional, Tuple

from research_api import OPERATIONS, RESULT_FIELDS, ResearchBackendError

logger = logging.getLogger("assistant.research.service")

DEFAULT_PORT = 8791
#: How often the client asks its local ctx whether Stop was pressed.
CANCEL_POLL_S = 0.25
#: Test/ops override, mirroring knowledge_service.ENV_EMBED_BASE: the name of a
#: module to install as `deep_research` in the SERVICE process before it serves.
#: The differential suite points both backends at one deterministic stand-in so
#: it can compare them without a live GPU, an LLM, or the open web. Ignored
#: unless set, so a normal service run uses the real pipeline.
ENV_DR_MODULE = "RESEARCH_SERVICE_DR_MODULE"

#: run_id -> _ServerCtx for runs currently in flight.
_RUNS: Dict[str, "_ServerCtx"] = {}
_RUNS_LOCK = threading.Lock()
#: Overrides patch deep_research's module globals, so one run at a time.
_RUN_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Server side
# --------------------------------------------------------------------------- #
class _ServerCtx:
    """The minimal context deep_research needs, owned by the SERVICE.

    deep_research touches ctx for exactly two things on this path: LLM calls
    (which go to LM Studio, already a separate process, configured from
    `config` in whichever process runs them) and `ctx.is_cancelled()`. This
    object supplies the second and lets the pipeline build the first from the
    service process's own config — which is why the client never has to send a
    ctx over the wire, and must not try to.
    """

    def __init__(self):
        self.cancel_event = threading.Event()
        self.gui_mode = False

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def set_stage(self, stage: str) -> None:
        """No-op: staging is a UI concern and the UI is in the other process.
        Defined because deep_research and its helpers call it unconditionally."""

    def remember(self, kind: str, text: str, meta: Optional[dict] = None) -> None:
        """No-op for the same reason: session memory belongs to the caller's
        turn, not to a stateless research run. See docs/gui_service_boundaries.md
        on why turn-local memory is deliberately NOT migrated."""


def _dispatch(op: str, args: Dict[str, Any]) -> Any:
    """Run one NON-STREAMING boundary operation. `run` is handled separately."""
    import deep_research as dr
    if op == "estimate_duration":
        secs, n = dr.estimate_duration(args.get("depth") or "standard")
        return [float(secs), int(n)]
    if op == "lang_of_text":
        return dr.lang_of_text(args.get("text") or "", args.get("default") or "en")
    if op == "knob_spec":
        return dict(dr.MANUAL_OVERRIDE_SPEC)
    if op == "knob_defaults":
        g = vars(dr)
        return {name: g.get(name) for name in dr.MANUAL_OVERRIDE_SPEC}
    if op == "cancel":
        return _cancel_run(str(args.get("run_id") or ""))
    raise ResearchBackendError(f"unknown operation: {op}")


def _cancel_run(run_id: str) -> bool:
    with _RUNS_LOCK:
        ctx = _RUNS.get(run_id)
    if ctx is None:
        return False
    ctx.cancel_event.set()
    return True


def _run_streaming(args: Dict[str, Any], emit: Callable[[Dict[str, Any]], None]) -> None:
    """Execute one research run, pushing NDJSON records through `emit`.

    Never raises: a failure is reported as an {"type": "error"} record so the
    client sees the same message it would have seen in-process, instead of a
    truncated stream it has to guess about.
    """
    import deep_research as dr
    ctx = _ServerCtx()
    run_id = uuid.uuid4().hex
    with _RUNS_LOCK:
        _RUNS[run_id] = ctx
    # First record, before any work: the client cannot cancel a run whose id it
    # does not have yet, so this must precede the pipeline, not follow it.
    emit({"type": "run_id", "run_id": run_id})
    topic = args.get("topic") or ""
    out_lang = args.get("out_lang")
    saved: Dict[str, Any] = {}
    try:
        with _RUN_LOCK:                       # module-global knobs: one at a time
            if out_lang is None:
                # research-service-boundary-v1 / out-lang-derived-from-typed-topic
                out_lang = dr.lang_of_text(topic)
            try:
                saved = dr.apply_overrides(args.get("overrides") or {}) \
                    if args.get("overrides") else {}
                result = dr.run_deep_research(
                    ctx, topic, depth=args.get("depth") or "standard",
                    out_lang=out_lang,
                    progress=lambda phase, stats, msg: emit(
                        {"type": "progress", "phase": phase,
                         "stats": stats, "msg": msg}),
                )
            finally:
                dr.restore_overrides(saved)
        emit({"type": "result", "result": result})
    except Exception as exc:
        logger.exception("research service run failed")
        emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        with _RUNS_LOCK:
            _RUNS.pop(run_id, None)


class _Handler(JsonHandler):
    server_version = "ResearchService/1"
    protocol_version = "HTTP/1.0"      # streaming body terminated by close

    def do_GET(self):                  # noqa: N802 (stdlib naming)
        if self.path.rstrip("/") in ("/health", ""):
            self._send(200, {"ok": True, "backend": "http", "ops": list(OPERATIONS)})
        else:
            self._send(404, {"ok": False, "error": f"no such path: {self.path}"})

    def do_POST(self):                 # noqa: N802
        path = self.path.rstrip("/")
        if path == "/shutdown":
            self._send(200, {"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if not path.startswith("/op/"):
            self._send(404, {"ok": False, "error": f"no such path: {self.path}"})
            return
        op = path[len("/op/"):]
        try:
            req = self._read_body()
        except Exception as exc:
            self._send(400, {"ok": False, "error": f"bad request: {exc}"})
            return
        args = req.get("args") or {}
        if op == "run":
            self._stream_run(args)
            return
        try:
            result = _dispatch(op, args)
        except Exception as exc:
            logger.exception("research service op %s failed", op)
            self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return
        self._send(200, {"ok": True, "result": result})

    def _stream_run(self, args: Dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()

        def emit(record: Dict[str, Any]) -> None:
            # Flushed per record: an unflushed progress tick is a progress tick
            # the user never sees, which is the whole reason this route streams.
            self.wfile.write((json.dumps(record, default=str) + "\n").encode("utf-8"))
            self.wfile.flush()

        _run_streaming(args, emit)


def make_server(port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Bind and return an unstarted server. Pass port=0 to let the OS pick one
    (read it back from srv.server_address[1])."""
    return ThreadingHTTPServer((host, port), _Handler)


def serve(port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> None:
    srv = make_server(port, host)
    msg = f"research service listening on http://{host}:{srv.server_address[1]}"
    logger.info("%s", msg)
    print(msg, flush=True)
    try:
        srv.serve_forever()
    finally:
        srv.server_close()


# --------------------------------------------------------------------------- #
# Client side
# --------------------------------------------------------------------------- #
class HttpResearchClient:
    """research_api.ResearchClient over HTTP. Same methods, same return shapes."""

    backend = "http"
    #: The server holds its own Event; Stop reaches it via POST /op/cancel keyed
    #: on the run id. Honest about costing a poll interval + a round-trip.
    cancellation_mode = "run_id"

    def __init__(self, url: str, *, timeout: float = 7200.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.last_run_id: Optional[str] = None

    # -- transport ---------------------------------------------------------- #
    def _post(self, op: str, **args) -> Any:
        import requests
        try:
            r = requests.post(f"{self.url}/op/{op}", json={"args": args}, timeout=60)
        except Exception as exc:
            raise ResearchBackendError(
                f"research service unreachable at {self.url}: {exc}") from exc
        try:
            body = r.json()
        except Exception as exc:
            raise ResearchBackendError(
                f"research service returned non-JSON ({r.status_code})") from exc
        if not body.get("ok"):
            raise ResearchBackendError(str(body.get("error") or f"HTTP {r.status_code}"))
        return body.get("result")

    def health(self) -> Dict[str, Any]:
        import requests
        return requests.get(f"{self.url}/health", timeout=10).json()

    # -- the run ------------------------------------------------------------ #
    def run(self, ctx, topic: str, *, depth: str = "standard",
            out_lang: Optional[str] = None,
            overrides: Optional[Dict[str, Any]] = None,
            progress: Optional[Callable[[str, dict, str], None]] = None) -> Dict[str, Any]:
        import requests
        args = {"topic": topic, "depth": depth, "out_lang": out_lang,
                "overrides": overrides or None}
        try:
            resp = requests.post(f"{self.url}/op/run", json={"args": args},
                                 stream=True, timeout=self.timeout)
        except Exception as exc:
            raise ResearchBackendError(
                f"research service unreachable at {self.url}: {exc}") from exc

        stop_watchdog = threading.Event()
        result: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        watchdog: Optional[threading.Thread] = None
        try:
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except Exception:
                    logger.debug("unparseable record from research service: %r", raw)
                    continue
                kind = rec.get("type")
                if kind == "run_id":
                    self.last_run_id = rec.get("run_id")
                    watchdog = self._start_watchdog(ctx, self.last_run_id, stop_watchdog)
                elif kind == "progress" and progress is not None:
                    # Re-invoked on the CALLER's side: a callable never crosses
                    # the boundary, only its arguments do.
                    progress(rec.get("phase") or "", rec.get("stats") or {},
                             rec.get("msg") or "")
                elif kind == "result":
                    result = rec.get("result")
                elif kind == "error":
                    error = rec.get("error")
        finally:
            stop_watchdog.set()
            if watchdog is not None:
                watchdog.join(timeout=2.0)
            resp.close()

        if error is not None:
            raise ResearchBackendError(error)
        if result is None:
            # A truncated stream is a failure, not an empty report: returning
            # {} here would let the GUI render "no sources found" for what was
            # actually a dropped connection.
            raise ResearchBackendError(
                "research service closed the stream without a result")
        return result

    def _start_watchdog(self, ctx, run_id: Optional[str],
                        stop: threading.Event) -> Optional[threading.Thread]:
        """Poll the CALLER's ctx for Stop and forward it to the service.

        This is what buys cancellation_mode == "run_id". Without it the Stop
        button would set a threading.Event the server can never see.
        """
        if run_id is None or ctx is None or not hasattr(ctx, "is_cancelled"):
            return None

        def _poll():
            while not stop.wait(CANCEL_POLL_S):
                try:
                    if ctx.is_cancelled():
                        self.cancel(ctx)
                        return
                except Exception:
                    logger.debug("cancel watchdog poll failed", exc_info=True)
                    return

        t = threading.Thread(target=_poll, name="research-cancel-watchdog", daemon=True)
        t.start()
        return t

    def cancel(self, ctx=None) -> bool:
        if not self.last_run_id:
            return False
        try:
            return bool(self._post("cancel", run_id=self.last_run_id))
        except Exception:
            logger.debug("cancel request failed", exc_info=True)
            return False

    # -- estimation / language ---------------------------------------------- #
    def estimate_duration(self, depth: str = "standard") -> Tuple[float, int]:
        secs, n = self._post("estimate_duration", depth=depth)
        return float(secs), int(n)

    def lang_of_text(self, text: str, default: str = "en") -> str:
        return str(self._post("lang_of_text", text=text, default=default))

    # -- knobs -------------------------------------------------------------- #
    def knob_spec(self) -> Dict[str, Any]:
        return self._post("knob_spec")

    def knob_defaults(self) -> Dict[str, Any]:
        return self._post("knob_defaults")


def result_is_complete(result: Dict[str, Any]) -> bool:
    """Every RESULT_FIELDS key present. Used by the differential suite to prove
    the two backends return the same SHAPE, not just the same report text."""
    return all(f in (result or {}) for f in RESULT_FIELDS)


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Standalone deep-research service.")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("RESEARCH_SERVICE_PORT", DEFAULT_PORT)))
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    install_dr_module(os.environ.get(ENV_DR_MODULE))
    serve(args.port, args.host)
    return 0


def install_dr_module(dotted: Optional[str]) -> bool:
    """Install `dotted` as this process's `deep_research`, before any request.

    Returns True if a substitution happened. Every server-side entry point
    resolves `deep_research` from sys.modules at call time, so this takes effect
    for every subsequent run without touching the request path.
    """
    if not dotted:
        return False
    import importlib
    import sys as _sys
    _sys.modules["deep_research"] = importlib.import_module(dotted)
    logger.info("research service using stand-in pipeline module: %s", dotted)
    return True


if __name__ == "__main__":
    raise SystemExit(main())
