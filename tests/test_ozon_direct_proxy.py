import socket
import threading
import time
from unittest.mock import MagicMock

import pytest

from services.ozon_direct_proxy import _Connect, _Server, _relay


def test_relay_preserves_backpressure_and_half_close():
    client, tunnel_in = socket.socketpair()
    tunnel_out, server = socket.socketpair()
    client.settimeout(10)
    server.settimeout(10)
    tunnel_out.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
    payload = b'x' * (5 * 1024 * 1024)
    errors = []

    def relay():
        try:
            _relay(tunnel_in, tunnel_out, timeout=8, initial=b'preface')
        except Exception as exc:
            errors.append(exc)

    def send():
        try:
            client.sendall(payload)
            client.shutdown(socket.SHUT_WR)
            assert client.recv(20) == b'ack'
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=relay, daemon=True)
    sender = threading.Thread(target=send, daemon=True)
    worker.start()
    sender.start()
    try:
        time.sleep(0.1)
        received = bytearray()
        while True:
            data = server.recv(65536)
            if not data:
                break
            received.extend(data)
        server.sendall(b'ack')
        server.shutdown(socket.SHUT_WR)
        sender.join(10)
        worker.join(10)
        assert not sender.is_alive() and not worker.is_alive()
        assert not errors and received == b'preface' + payload
    finally:
        for sock in (client, tunnel_in, tunnel_out, server):
            sock.close()


def test_incomplete_oversized_connect_never_establishes_a_tunnel():
    with _Server(('127.0.0.1', 0), _Connect) as proxy:
        proxy.ifindex = 0
        worker = threading.Thread(target=proxy.serve_forever, daemon=True)
        worker.start()
        try:
            with socket.create_connection(proxy.server_address, timeout=3) as client:
                client.sendall(b'CONNECT ozon.ru:443 HTTP/1.1\r\nX: ' + b'x' * 9000)
                client.shutdown(socket.SHUT_WR)
                assert not client.recv(1024).startswith(b'HTTP/1.1 200')
        finally:
            proxy.shutdown()
            worker.join(3)


def test_relay_error_identifies_the_socket_side(monkeypatch, caplog):
    client, remote = MagicMock(), MagicMock()
    client.recv.side_effect = ConnectionAbortedError(10053, 'closed')
    monkeypatch.setattr('services.ozon_direct_proxy.select.select', lambda *args: ([client], [], []))
    with pytest.raises(ConnectionAbortedError):
        _relay(client, remote)
    assert 'read failed on browser side' in caplog.text
