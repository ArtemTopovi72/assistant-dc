"""EasyOCR in its own process (see ocr_reader._WorkerReader).

Inside the app it took 107 s on a picture this process reads in ~4 s (live
2026-09-27): its torch pool fought ~200 threads of Whisper/F5/reranker/Qt and
the GUI looked hung. One JSON line in ({"path": ...}), one JSON line out.
"""
import json
import sys


def main() -> None:
    import easyocr
    reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            res = reader.readtext(req["path"], detail=1, paragraph=False)
            out = [[[[float(x), float(y)] for x, y in box], str(text), float(conf)]
                   for box, text, conf in res]
            # ASCII escapes: a piped stdout on Windows is cp1251 and «ы» (0xfb) broke the reader.
            print(json.dumps({"ok": out}), flush=True)
        except Exception as exc:
            print(json.dumps({"error": repr(exc)}), flush=True)


if __name__ == "__main__":
    main()
