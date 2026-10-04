"""PaddleOCR (PP-OCRv5 East-Slavic) over the pages of a PDF. Runs in venv_paddle (paddlepaddle CPU + paddleocr + pypdfium2).

    venv_paddle/Scripts/python scripts/paddle_pdf_worker.py <pdf> [max_pages]
One JSON line per page on stdout: {"page": 1, "total": N, "items": [[poly, text, score], ...]}
"""
import json
import os
import sys
import tempfile


def main() -> None:
    pdf = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    import pypdfium2
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="ru", enable_mkldnn=False, use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False)
    doc = pypdfium2.PdfDocument(pdf)
    total = min(len(doc), cap)
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(total):
            png = os.path.join(tmp, f"p{i}.png")
            doc[i].render(scale=2.0).to_pil().convert("RGB").save(png)
            items = []
            for r in ocr.predict(png):
                d = r.json["res"]
                for poly, text, score in zip(d.get("rec_polys", []), d.get("rec_texts", []), d.get("rec_scores", [])):
                    items.append([[[float(x), float(y)] for x, y in poly], str(text), float(score)])
            os.unlink(png)
            print(json.dumps({"page": i + 1, "total": total, "items": items}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
