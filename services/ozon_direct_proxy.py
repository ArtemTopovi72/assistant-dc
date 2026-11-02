"""Loopback CONNECT proxy that sends only Ozon browser traffic over one interface.

On Windows Red Shield routes the default browser through its VPN exit, which Ozon
rejects. IP_UNICAST_IF selects the physical interface for this one browser without
changing system routes or disconnecting the VPN for other applications.
"""
import select
import socket
import socketserver
import struct
import threading


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _Connect(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(20)
        head = b""
        while b"\r\n\r\n" not in head and len(head) < 8192:
            try:
                part = self.request.recv(1024)
            except OSError:  # Chromium keeps idle proxy sockets around.
                return
            if not part:
                return
            head += part
        fields = head.split(b"\r\n", 1)[0].decode("ascii", "replace").split()
        if len(fields) != 3 or fields[0] != "CONNECT":
            self.request.sendall(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
            return
        try:
            host, port = fields[1].lower().rsplit(":", 1)
            allowed = host in ("ozon.ru", "ozone.ru") or host.endswith((".ozon.ru", ".ozone.ru"))
            if not allowed or port != "443":
                self.request.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
                return
            remote = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            remote.settimeout(20)
            remote.setsockopt(socket.IPPROTO_IP, 31, struct.pack("!I", self.server.ifindex))
            try:
                remote.connect((host, 443))
                self.request.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                remote.setblocking(False)
                self.request.setblocking(False)
                while True:
                    readable, _, _ = select.select((self.request, remote), (), (), 30)
                    if not readable:
                        return
                    for source in readable:
                        data = source.recv(65536)
                        if not data:
                            return
                        (remote if source is self.request else self.request).sendall(data)
            finally:
                remote.close()
        except (OSError, ValueError):
            return


_server = None
_lock = threading.Lock()


def start(ifindex: int) -> str:
    """Return the proxy URL. The server is loopback-only and lives with the bot."""
    global _server
    with _lock:
        if _server is None:
            _server = _Server(("127.0.0.1", 0), _Connect)
            _server.ifindex = int(ifindex)
            threading.Thread(target=_server.serve_forever, name="ozon-direct-proxy",
                             daemon=True).start()
        elif _server.ifindex != int(ifindex):
            raise ValueError("Ozon direct interface changed while the browser is running")
        return f"http://127.0.0.1:{_server.server_address[1]}"
