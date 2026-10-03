"""The JSON envelope shared by the out-of-process services.

knowledge_service and research_service are deliberately parallel: same
`/health`, same `{"ok": ...}` body shape, same stdlib-only rule.  They also
carried a byte-identical response writer, which is the kind of duplication that
stays harmless right up to the day one of them gains a header the other lacks
and the two backends start disagreeing about a detail no test names.

Stdlib only, on purpose -- both services exist so they can start in a bare
interpreter, without PyQt5 or torch, and a shared base must not take that away.
"""
import json
import logging
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict

logger = logging.getLogger(__name__)


class JsonHandler(BaseHTTPRequestHandler):
    """Response writing and request reading; routing stays in each service."""

    def log_message(self, fmt, *a):        # keep the console clean
        logger.debug("%s - %s", self.address_string(), fmt % a)

    def _send(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")
