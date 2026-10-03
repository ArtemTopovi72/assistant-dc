"""A fresh clone has no runtime/ (gitignored): importing config must create it."""
import importlib, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))


def test_config_creates_output_dir():
    d = Path(tempfile.mkdtemp()) / "fresh" / "runtime"
    os.environ["ASSISTANT_OUTPUT_DIR"] = str(d)
    try:
        import config
        importlib.reload(config)
        assert d.is_dir(), d
    finally:
        del os.environ["ASSISTANT_OUTPUT_DIR"]
        importlib.reload(config)


if __name__ == "__main__":
    test_config_creates_output_dir()
    print("ok")
