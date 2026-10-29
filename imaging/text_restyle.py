"""Restyle recorded headline captions without repainting the rest of an image."""
import difflib
import os
import re
import time
from pathlib import Path

from seams import dep


def _words(text):
    text = str(text).upper().translate(str.maketrans('ABCEHKMOPTXY', 'АВСЕНКМОРТХУ'))
    return re.findall(r'[A-ZА-ЯЁ0-9]+', text)


def caption_regions(regions, layout, width, height):
    top, bottom = [], []
    for item in layout.get('elements', []):
        text = str(item.get('text') or '').strip().replace('\n', ' ')
        desc = str(item.get('desc') or '')
        if re.search(r'shirt|clothing|dress|jacket|fabric|одежд|футболк', desc, re.I):
            continue
        if text and not re.search(r'(?:text|lettering|caption).*(?:across|top|bottom|image|picture)', desc, re.I):
            return [], [], []
        y = float(item.get('y') or 0) + float(item.get('h') or 0) / 2
        if text and y < .25:
            top.append(text)
        elif text and y > .75:
            bottom.append(text)
        elif text:
            return [], [], []
    expected = [word for text in top + bottom for word in _words(text)]
    if not expected:
        return [], top, bottom
    selected, found = [], set()
    for points, text, confidence in regions:
        if confidence < .35:
            continue
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        centre = (min(ys) + max(ys)) / (2 * height)
        if not (centre < .25 or centre > .75):
            continue
        words = _words(text)
        matches = {i for i, target in enumerate(expected) if any(
            difflib.SequenceMatcher(None, w, target).ratio() >= .78 for w in words)}
        if matches:
            selected.append((max(0, int(min(xs))), max(0, int(min(ys))),
                             min(width, int(max(xs))), min(height, int(max(ys)))))
            found |= matches
    return (selected if len(found) == len(expected) else []), top, bottom


def caption_left(regions, boxes, captions):
    expected = [word for text in captions for word in _words(text)]
    for points, text, confidence in regions:
        if confidence < .35:
            continue
        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)
        if not any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in boxes):
            continue
        if any(difflib.SequenceMatcher(None, word, target).ratio() >= .78
               for word in _words(text) for target in expected):
            return True
    return False


def protected_boxes(layout, width, height):
    out = []
    for item in layout.get('elements', []):
        desc = str(item.get('desc') or '')
        if not (re.search(r'logo|логотип', desc, re.I) or
                (item.get('text') and re.search(r'shirt|clothing|dress|jacket|одежд|футболк', desc, re.I))):
            continue
        try:
            x, y, w, h = (float(item[k]) for k in ('x', 'y', 'w', 'h'))
            box = (max(0, int(x * width)), max(0, int(y * height)),
                   min(width - 1, int((x + w) * width)), min(height - 1, int((y + h) * height)))
            out.append(box)
        except (ValueError, TypeError, KeyError, OverflowError):
            continue
    return out


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def restyle(ctx, source, layout_record):
    import image_lettering_remove as remove
    import ocr_reader
    from PIL import Image, ImageDraw, ImageFont
    import image
    reader = ocr_reader._get()
    if reader is None:
        return None
    with Image.open(source) as opened:
        original = opened.convert('RGB')
    boxes, top, bottom = caption_regions(reader.readtext(source, detail=1, paragraph=False),
                                          layout_record['layout'], *original.size)
    if not boxes:
        return None
    width, height = original.size
    protected = remove._faces(source) + protected_boxes(layout_record['layout'], width, height)
    if any(overlaps(box, keep) for box in boxes for keep in protected):
        return None
    root = Path(image.OUTPUT_DIR)
    root.mkdir(parents=True, exist_ok=True)
    top_boxes = [b for b in boxes if (b[1] + b[3]) / 2 < height / 2]
    bottom_boxes = [b for b in boxes if (b[1] + b[3]) / 2 >= height / 2]
    result = original
    current = source
    for group_boxes, captions in ((top_boxes, top), (bottom_boxes, bottom)):
        if not group_boxes:
            continue
        mask = Image.new('L', original.size, 0)
        md = ImageDraw.Draw(mask)
        for x0, y0, x1, y1 in group_boxes:
            md.rectangle((max(0, x0 - 3), max(0, y0 - 3), min(width - 1, x1 + 3), min(height - 1, y1 + 3)), fill=255)
        mask_path = str(root / f'_INTERMEDIATE_caption_mask_{time.time_ns()}.png')
        mask.save(mask_path)
        safe_mask = mask.copy()
        safe_draw = ImageDraw.Draw(safe_mask)
        for face in protected:
            safe_draw.rectangle(face, fill=0)
        clean = dep(ctx, 'fill_caption', remove._fill_objectclear)(ctx, current, mask_path, 10)
        if not clean or not os.path.exists(clean) or caption_left(
                reader.readtext(clean, detail=1, paragraph=False), group_boxes, captions):
            clean = dep(ctx, 'fill_caption_fallback', remove._fill_firered)(ctx, current, mask_path, 10)
        if not clean or not os.path.exists(clean):
            return None
        with Image.open(clean) as edited:
            if edited.size != original.size:
                return None
            result = Image.composite(edited.convert('RGB'), result, safe_mask)
        check_path = root / f'_INTERMEDIATE_caption_check_{time.time_ns()}.png'
        result.save(check_path)
        if caption_left(reader.readtext(str(check_path), detail=1, paragraph=False), group_boxes, captions):
            check_path.unlink(missing_ok=True)
            return None
        current = str(check_path)
    draw = ImageDraw.Draw(result)
    def bounds(group):
        return (min(b[0] for b in group), min(b[1] for b in group),
                max(b[2] for b in group), max(b[3] for b in group)) if group else None
    areas = [(top, bounds(top_boxes)), (bottom, bounds(bottom_boxes))]
    if any(lines and not area for lines, area in areas):
        return None
    size = max(12, int(height * .055))
    while True:
        try:
            font = ImageFont.truetype('arial.ttf', size)
        except OSError:
            font = ImageFont.truetype('DejaVuSans.ttf', size)
        spacing = int(size * 1.28)
        fits = all(not lines or (max(draw.textlength(line, font=font) for line in lines) + 4 <= area[2] - area[0]
                                and spacing * len(lines) + 4 <= area[3] - area[1])
                   for lines, area in areas)
        if fits or size <= 12:
            break
        size -= 1
    if not fits:
        return None
    placements = []
    for group, area in areas:
        if not group:
            continue
        y = area[1] + (area[3] - area[1] - spacing * len(group)) // 2
        for line in group:
            x = area[0] + (area[2] - area[0] - draw.textlength(line, font=font)) / 2
            bounds = draw.textbbox((x, y), line, font=font, stroke_width=2)
            if any(overlaps(bounds, box) for box in protected):
                return None
            placements.append((line, x, y))
            y += spacing
    for line, x, y in placements:
        draw.text((x + 1, y + 2), line, font=font, fill=(15, 15, 15), stroke_width=1)
        draw.text((x, y), line, font=font, fill='white')
    out = root / f'text_restyle_{time.time_ns()}.png'
    result.save(out)
    return str(out)
