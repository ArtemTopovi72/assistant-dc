"""Local servers are reached directly even when a proxy is configured.

With HTTP_PROXY set and no NO_PROXY, requests sent 127.0.0.1 through the
proxy and every call to LM Studio / ComfyUI failed while they ran fine.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CHILD = r"""
import sys; sys.path.insert(0, %r)
import config, requests.utils as u, os
print(os.environ["NO_PROXY"])
print(u.should_bypass_proxies("http://127.0.0.1:1234/v1/models", None),
      u.should_bypass_proxies("http://localhost:8000/", None),
      u.should_bypass_proxies("https://example.com/", None))
""" % os.path.join(ROOT, "core")


def _run(env_extra):
    env = {k: v for k, v in os.environ.items() if k.lower() not in ("no_proxy",)}
    env.update(env_extra, HTTP_PROXY="http://10.255.255.1:3128", HTTPS_PROXY="http://10.255.255.1:3128")
    out = subprocess.run([sys.executable, "-c", _CHILD], env=env, capture_output=True,
                         text=True, timeout=120).stdout.split("\n")
    return out[0], out[1]


def test_loopback_bypasses_the_proxy():
    no_proxy, verdict = _run({})
    assert "127.0.0.1" in no_proxy and "localhost" in no_proxy
    assert verdict == "True True False", verdict


def test_user_entries_are_kept():
    no_proxy, verdict = _run({"NO_PROXY": "corp.local", "no_proxy": "corp.local"})
    assert no_proxy.startswith("corp.local,") and "127.0.0.1" in no_proxy
    assert verdict == "True True False", verdict
