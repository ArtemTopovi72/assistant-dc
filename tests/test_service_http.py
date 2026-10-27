"""Malformed local RPC requests fail before an unbounded body read."""
import io

import pytest

from service_http import JsonHandler


def _handler(length, body):
    handler = object.__new__(JsonHandler)
    handler.headers = {"Content-Length": length}
    handler.rfile = io.BytesIO(body)
    return handler


@pytest.mark.parametrize("length", ["-1", "67108865", "not-a-number"])
def test_invalid_length_is_rejected_before_reading(length):
    handler = _handler(length, b"{}")
    with pytest.raises(ValueError):
        handler._read_body()
    assert handler.rfile.tell() == 0


@pytest.mark.parametrize("body", [b"[]", b"null", b"1", b'"text"'])
def test_rpc_body_must_be_a_json_object(body):
    handler = _handler(str(len(body)), body)
    with pytest.raises(ValueError):
        handler._read_body()


def test_incomplete_request_is_rejected():
    with pytest.raises(ValueError):
        _handler("5", b"{}")._read_body()


def test_valid_json_and_empty_requests_remain_accepted():
    assert _handler("12", b'{"value": 1}')._read_body() == {"value": 1}
    assert _handler("0", b"")._read_body() == {}
