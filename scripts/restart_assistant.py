"""Relaunch Assistant DC after its singleton lock has been released."""
import ctypes
import os
import subprocess
import sys


def main():
    pid = int(sys.argv[1])
    if os.name == "nt":
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if handle:
            try:
                if kernel.WaitForSingleObject(handle, 120_000) != 0:
                    return  # Still running: never start a second bot.
            finally:
                kernel.CloseHandle(handle)
    else:
        import time
        for _ in range(120):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(1)
        else:
            return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    launcher = os.path.join(root, "scripts", "launch_all.py")
    venv_python = os.path.join(root, "venv", "Scripts", "pythonw.exe") if os.name == "nt" else os.path.join(root, "venv", "bin", "python")
    python = venv_python if os.path.isfile(venv_python) else sys.executable
    subprocess.Popen([python, launcher], cwd=root,
                     creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
