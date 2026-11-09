"""Loopback CONNECT proxy that sends only Ozon browser traffic over one interface.

On Windows Red Shield routes the default browser through its VPN exit, which Ozon
rejects. IP_UNICAST_IF selects the physical interface for this one browser without
changing system routes or disconnecting the VPN for other applications.
"""
import select
import os
import logging
import socket
import socketserver
import struct
import subprocess
import threading

logger = logging.getLogger("assistant.ozon.direct_proxy")


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _relay(client, remote, timeout=30, initial=b''):
    peers = {client: remote, remote: client}
    buffers = {client: bytearray(), remote: bytearray()}
    buffers[remote].extend(initial)
    readable = {client, remote}
    shutdown = set()
    for sock in peers:
        sock.setblocking(False)
    while readable or any(buffers.values()):
        for source in peers:
            dest = peers[source]
            if source not in readable and not buffers[dest] and dest not in shutdown:
                try:
                    dest.shutdown(socket.SHUT_WR)
                except OSError:
                    logger.warning("Ozon relay half-close failed on %s side", "browser" if dest is client else "remote")
                    raise
                shutdown.add(dest)
        reads = [source for source in readable if len(buffers[peers[source]]) < 262144]
        writes = [dest for dest, pending in buffers.items() if pending]
        try:
            ready_read, ready_write, _ = select.select(reads, writes, (), timeout)
        except OSError:
            logger.warning("Ozon relay select failed")
            raise
        if not ready_read and not ready_write:
            return
        for source in ready_read:
            try:
                data = source.recv(65536)
            except BlockingIOError:
                continue
            except OSError:
                logger.warning("Ozon relay read failed on %s side", "browser" if source is client else "remote")
                raise
            if data:
                buffers[peers[source]].extend(data)
            else:
                readable.remove(source)
        for dest in ready_write:
            try:
                sent = dest.send(buffers[dest])
            except BlockingIOError:
                continue
            except OSError:
                logger.warning("Ozon relay write failed on %s side", "browser" if dest is client else "remote")
                raise
            if not sent:
                return
            del buffers[dest][:sent]


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
        if b"\r\n\r\n" not in head:
            return
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
                logger.debug("Ozon direct tunnel connected to %s via interface %s", host,
                             self.server.ifindex)
                self.request.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                _relay(self.request, remote, initial=head.split(b"\r\n\r\n", 1)[1])
            finally:
                remote.close()
        except (OSError, ValueError) as exc:
            logger.warning("Ozon direct tunnel to %s failed: %s", fields[1] if len(fields) > 1 else "?", exc)
            return


_server = None
_lock = threading.Lock()


def adapter_index(name: str) -> int:
    """Resolve a stable Windows adapter name after VPN reconnects renumber it."""
    env = dict(os.environ, OZON_DIRECT_ADAPTER_NAME=name)
    value = subprocess.check_output(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command",
         "(Get-NetAdapter -Name $env:OZON_DIRECT_ADAPTER_NAME -ErrorAction Stop).ifIndex"],
        env=env, text=True, timeout=10).strip()
    return int(value)


def start(ifindex: int) -> str:
    """Return the proxy URL. The server is loopback-only and lives with the bot."""
    global _server
    with _lock:
        if _server is None:
            _server = _Server(("127.0.0.1", 0), _Connect)
            _server.ifindex = int(ifindex)
            threading.Thread(target=_server.serve_forever, name="ozon-direct-proxy",
                             daemon=True).start()
        else:
            _server.ifindex = int(ifindex)
        return f"http://127.0.0.1:{_server.server_address[1]}"
