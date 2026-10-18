"""The annotator's reply becomes Klein's prompt; only rows that really change feed the pixel lock."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import klein_annotate as ka

reply = ('Move the boat <boat_1> to the right. Everything else stays identical. '
         '[{"id":"boat_1","from":"ref_image_0","src_bbox":[400,150,520,350],"tgt_bbox":[400,600,520,800],"desc":"boat"},'
         '{"id":"buoy_1","from":"ref_image_0","src_bbox":[600,400,680,440],"tgt_bbox":[600,400,680,440],"desc":"buoy"}]')
para, rows = ka.parse_record(reply)
assert para.startswith("Move the boat") and len(rows) == 2
assert ka.edit_boxes(rows) == [[400, 150, 520, 350], [400, 600, 520, 800]]
assert ka.parse_record("just words") is None and ka.parse_record("x [{bad") is None


class Ctx:
    vision_model = staticmethod(lambda **kw: reply)


r = ka.annotate_edit(Ctx(), "x.png", "move the boat right")
assert r and r["prompt"].startswith("Move the boat") and len(r["boxes"]) == 2


Ctx.vision_model = staticmethod(lambda **kw: "no table")
assert ka.annotate_edit(Ctx(), "x.png", "i") is None
print("ok")
