"""PaddleOCR (PP-OCRv5, ru) as ocr_reader's reader process -- the same protocol as
imaging/ocr_worker.py: one JSON line in ({"path": ...}), one out ({"ok": [[box, text, score], ...]}).
Runs in venv_paddle (paddlepaddle CPU + paddleocr), like paddle_pdf_worker.py: it reads
Russian print nearly error-free where EasyOCR misses letters.
"""
import json
import sys


def main() -> None:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="ru", enable_mkldnn=False, use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            out = []
            for r in ocr.predict(req["path"]):
                d = r.json["res"]
                for poly, text, score in zip(d.get("rec_polys", []), d.get("rec_texts", []),
                                             d.get("rec_scores", [])):
                    out.append([[[float(x), float(y)] for x, y in poly], str(text), float(score)])
            # ASCII escapes: a piped stdout on Windows is cp1251 (see ocr_worker.py).
            print(json.dumps({"ok": out}), flush=True)
        except Exception as exc:
            print(json.dumps({"error": repr(exc)}), flush=True)


if __name__ == "__main__":
    main()
