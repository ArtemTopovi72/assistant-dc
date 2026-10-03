"""config.py must not import Torch just to be imported.

`import torch` costs ~1.9 s and ~430 MB RSS. config.py is imported by 80+
modules and by every one of these standalone suites, and almost none of them
ever read a device string — so DEVICE / WHISPER_DEVICE / WHISPER_COMPUTE_TYPE
are resolved lazily via a module __getattr__ instead of at import time.

This suite pins the whole contract: the import stays Torch-free, each name still
resolves to the same value it always did, `from config import DEVICE` works
(module __getattr__ is consulted by the from-import machinery too), the answer is
memoized after the first probe, an unknown attribute still raises AttributeError,
and a broken/absent Torch degrades to the CPU settings instead of exploding at
import time.

Offline and deterministic: the only subprocesses are this same interpreter
importing local modules. No GPU work, no network.
Run: venv/Scripts/python.exe tests/test_config_lazy_device.py
"""
import os, subprocess, sys, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


def _run(body: str):
    """Run `body` in a fresh interpreter rooted at the project; return stdout."""
    src = "import sys; sys.path.insert(0, r'%s')\n" % ROOT + textwrap.dedent(body)
    p = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, cwd=ROOT)
    if p.returncode != 0:
        return "SUBPROCESS-FAILED: " + (p.stderr or "")[-400:]
    return p.stdout.strip()


def test_import_does_not_pull_torch():
    out = _run("""
        import config
        print('torch' in sys.modules)
    """)
    check("import_config_leaves_torch_unloaded", out == "False", out)


def test_importing_lane_modules_does_not_pull_torch():
    # The modules that only ever needed config for its plain settings must stay
    # Torch-free too — audio.py in particular used to name DEVICE in its
    # top-level `from config import ...`, which dragged Torch into every process.
    for mod in ("audio", "llm", "tools", "music", "comfy_client"):
        out = _run("import %s\nprint('torch' in sys.modules)" % mod)
        check("import_%s_leaves_torch_unloaded" % mod, out == "False", out)


def test_device_names_resolve_and_agree():
    out = _run("""
        import config
        cuda = False
        try:
            import torch
            cuda = bool(torch.cuda.is_available())
        except Exception:
            pass
        expect = ('cuda:0' if cuda else 'cpu',
                  'cuda' if cuda else 'cpu',
                  'int8_float16' if cuda else 'int8')
        got = (config.DEVICE, config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE)
        print(got == expect, got)
    """)
    check("device_names_match_torch_reality", out.startswith("True"), out)


def test_from_import_still_works():
    out = _run("""
        from config import DEVICE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
        print(DEVICE in ('cuda:0', 'cpu')
              and WHISPER_DEVICE in ('cuda', 'cpu')
              and WHISPER_COMPUTE_TYPE in ('int8_float16', 'int8'))
    """)
    check("from_config_import_DEVICE_works", out == "True", out)


def test_probe_is_memoized():
    # One access must bake ALL THREE names into the module globals, so Torch is
    # probed exactly once no matter how often a hot loop reads config.DEVICE.
    out = _run("""
        import config
        config.DEVICE
        print(all(n in vars(config) for n in
                  ('DEVICE', 'WHISPER_DEVICE', 'WHISPER_COMPUTE_TYPE')))
    """)
    check("first_access_memoizes_all_three", out == "True", out)


def test_unknown_attribute_still_raises():
    out = _run("""
        import config
        try:
            config.DEFINITELY_NOT_A_SETTING
            print('NO-RAISE')
        except AttributeError as e:
            print('AttributeError' if 'DEFINITELY_NOT_A_SETTING' in str(e) else 'BAD-MSG')
    """)
    check("unknown_attr_raises_AttributeError", out == "AttributeError", out)


def test_broken_torch_degrades_to_cpu():
    # A Torch that raises on import (the CPU/CUDA build swap, a half-installed
    # wheel) must not take config down — it degrades to the CPU settings.
    out = _run("""
        class _Boom:
            def find_module(self, name, path=None): return None
            def find_spec(self, name, target=None, path=None):
                if name == 'torch' or name.startswith('torch.'):
                    raise ImportError('simulated broken torch')
                return None
        sys.meta_path.insert(0, _Boom())
        sys.modules.pop('torch', None)
        import config
        print(config.DEVICE, config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE)
    """)
    check("broken_torch_falls_back_to_cpu", out == "cpu cpu int8", out)


def test_dir_lists_lazy_names():
    out = _run("""
        import config
        d = dir(config)
        print(all(n in d for n in ('DEVICE', 'WHISPER_DEVICE', 'WHISPER_COMPUTE_TYPE'))
              and 'MODEL_NAME' in d)
    """)
    check("dir_exposes_lazy_names", out == "True", out)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print("  ERROR in " + fn.__name__ + ": " + type(e).__name__ + ": " + str(e))
    passed = sum(1 for _, c in RESULTS if c)
    print(f"\n{len(fns)-failed}/{len(fns)} functions, {passed}/{len(RESULTS)} checks passed")
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)
