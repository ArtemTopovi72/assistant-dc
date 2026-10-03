"""Drive code_runner's Docker backend against a REAL engine.

Everything in code_runner's container path -- the read-only root, --network
none, the memory and pid ceilings, the tmpfs, PYTHONPATH into the sandbox's
own site-packages -- has never once executed, because the engine on this
machine could not start (see the WSL virtual-machine-platform diagnosis). The
unit suites cover the selection logic with docker_available() stubbed, which
proves which branch is taken and nothing about what the branch does.

This lives in bench/, not tests/: it needs a live Docker daemon and pulls an
image over the network. Nothing here may be imported by the suite.

    venv\\Scripts\\python.exe bench\\sandbox_docker_live.py

Exit 0 = every claim in backend_status() held up. Exit 2 = no engine, so
nothing was measured (which is NOT a pass).
"""
import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import code_runner as CR
from code_sandbox import Sandbox

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name
          + ("" if cond or not detail else "  -- " + str(detail)[:400]))


def _sandbox():
    return Sandbox(tempfile.mkdtemp(prefix="sbx_docker_"))


def main() -> int:
    exe = CR.docker_binary()
    if not exe:
        print("CANNOT RUN: no docker binary on this machine.")
        return 2
    if not CR.docker_available(timeout=30):
        print("CANNOT RUN: docker.exe is present but the engine does not answer "
              "`docker info`. Start Docker Desktop and run this again.")
        return 2

    backend, why = CR.backend_status()
    check("backend_is_docker", backend == "docker", why)
    print("    " + why)

    # The image must exist before the timing claims mean anything; the first
    # run would otherwise measure a pull.
    print("... pulling " + CR.CONTAINER_IMAGE + " (first run only)")
    import subprocess
    subprocess.run([exe, "pull", CR.CONTAINER_IMAGE], capture_output=True,
                   text=True, timeout=900)

    sb = _sandbox()

    # 1. it runs at all, and reports the container backend
    r = CR.run_python(sb, "print('hello from', __import__('platform').system())")
    check("runs_and_reports_docker", r.ok and r.backend == "docker", r.output)
    check("runs_on_linux_not_windows", "Linux" in (r.output or ""), r.output)

    # 2. --network none is real, not a flag we merely pass
    net = CR.run_python(sb, "\n".join([
        "import socket",
        "socket.setdefaulttimeout(6)",
        "try:",
        "    socket.create_connection(('1.1.1.1', 53))",
        "    print('NETWORK REACHABLE')",
        "except Exception as e:",
        "    print('blocked:', type(e).__name__)",
    ]), timeout=40)
    check("script_run_has_no_network",
          "NETWORK REACHABLE" not in (net.output or ""), net.output)

    # 3. the container root really is read-only, and /tmp really is writable
    ro = CR.run_python(sb, "\n".join([
        "try:",
        "    open('/usr/lib/proof.txt', 'w').write('x')",
        "    print('ROOT WRITABLE')",
        "except Exception as e:",
        "    print('root read-only:', type(e).__name__)",
        "open('/tmp/ok.txt', 'w').write('x')",
        "print('tmp writable')",
    ]))
    check("container_root_is_read_only",
          "ROOT WRITABLE" not in (ro.output or ""), ro.output)
    check("tmpfs_is_writable", "tmp writable" in (ro.output or ""), ro.output)

    # 4. the sandbox is the ONLY durable surface, and it is shared with the host
    w = CR.run_python(sb, "open('made_in_container.txt','w').write('written')")
    check("sandbox_is_writable", w.ok, w.output)
    check("sandbox_write_reaches_the_host",
          (sb.root / "made_in_container.txt").exists(), sorted(
              p.name for p in sb.root.iterdir()))

    # 5. the memory ceiling is enforced by the engine, not by hope.
    #    Both halves are needed. "did not print ALLOCATED 8GB" alone is
    #    satisfied by ANY failure -- a missing image, a typo, a dead daemon --
    #    so it has to be paired with a run just under the cap that succeeds,
    #    or the check passes for reasons that have nothing to do with memory.
    under = CR.run_python(
        sb, "b = bytearray(1024 * 1024 * 1024); print('1GB ok', len(b))",
        timeout=120)
    check("allocation_under_the_cap_succeeds", under.ok and "1GB ok" in (under.output or ""),
          "rc=%s out=%s" % (under.code, under.output))
    mem = CR.run_python(sb, "\n".join([
        "buf = bytearray()",
        "try:",
        "    for _ in range(64):",
        "        buf += bytearray(128 * 1024 * 1024)",   # up to 8 GB, cap is 2g
        "    print('ALLOCATED 8GB')",
        "except MemoryError:",
        "    print('MemoryError')",
    ]), timeout=120)
    # 137 = 128 + SIGKILL: the engine's OOM killer, not Python noticing.
    check("memory_ceiling_holds",
          (not mem.ok) and "ALLOCATED 8GB" not in (mem.output or ""),
          "rc=%s out=%s" % (mem.code, mem.output))
    check("over_cap_is_oom_killed", mem.code == 137,
          "expected exit 137 (SIGKILL), got %s" % mem.code)

    # 5b. the pid ceiling. Same shape: a run that stays under it must succeed,
    #     or "the big one failed" proves nothing. The fork storm is safe to
    #     write here precisely BECAUSE the limit is what this checks -- if it
    #     is not enforced, the container takes 256+ threads and the check says
    #     so rather than the host paying for it.
    few = CR.run_python(sb, "\n".join([
        "import threading, time",
        "ts = [threading.Thread(target=time.sleep, args=(2,)) for _ in range(32)]",
        "[t.start() for t in ts]",
        "[t.join() for t in ts]",
        "print('32 threads ok')",
    ]), timeout=90)
    check("threads_under_the_pid_cap_run", few.ok and "32 threads ok" in (few.output or ""),
          "rc=%s out=%s" % (few.code, few.output))
    many = CR.run_python(sb, "\n".join([
        "import threading, time",
        "started = 0",
        "try:",
        "    for _ in range(2000):",
        "        threading.Thread(target=time.sleep, args=(30,), daemon=True).start()",
        "        started += 1",
        "    print('STARTED ALL', started)",
        "except Exception as e:",
        "    print('stopped at', started, type(e).__name__)",
    ]), timeout=90)
    check("pid_ceiling_holds", "STARTED ALL" not in (many.output or ""),
          "rc=%s out=%s" % (many.code, many.output))

    # 6. the deadline kills the tree, and does so near the deadline
    t0 = time.monotonic()
    slow = CR.run_python(sb, "import time; time.sleep(120); print('SLEPT')",
                         timeout=8)
    dt = time.monotonic() - t0
    check("timeout_kills_the_run", not slow.ok and "SLEPT" not in (slow.output or ""),
          slow.output)
    check("timeout_is_honoured_promptly", dt < 45, "%.1fs for an 8s deadline" % dt)

    # 7. install() is the one call given a network, and what it installs is
    #    importable afterwards by a run that has none.
    ins = CR.install(sb, ["six"], timeout=300)
    check("install_succeeds_with_network", ins.ok, ins.output)
    use = CR.run_python(sb, "import six; print('six', six.__version__)")
    check("installed_package_importable_offline",
          use.ok and "six" in (use.output or ""), use.output)

    ok = sum(1 for _, c in RESULTS if c)
    print("\n%d/%d checks passed" % (ok, len(RESULTS)))
    bad = [n for n, c in RESULTS if not c]
    if bad:
        print("FAILED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
