"""bench/context_cost + context_acon: the counting and the rule merge."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
import context_cost as CC
import context_acon as CA


def test_cost_counts_repeats_reasks_recalls():
    call = lambda i, n, a: {"role": "assistant", "content": "", "tool_calls": [
        {"id": i, "function": {"name": n, "arguments": a}}]}
    h = [{"role": "user", "content": "нарисуй кота"}, call("1", "generate_image", '{"d":"cat"}'),
         {"role": "tool", "tool_call_id": "1", "content": "ok"},
         {"role": "user", "content": "убери фон"},
         {"role": "assistant", "content": "О какой картинке речь?"},
         call("2", "generate_image", '{"d":"cat"}'), call("3", "recall_context", '{"id":"a1234567"}')]
    s = CC.score_history(h)
    assert s["repeat_calls"] == 1 and s["reasks"] == 1 and s["recalls"] == 1 and s["turns"] == 2


def test_acon_merge_dedupes_caps_and_refuses_keep_all():
    rules = CA.merge(["Keep picture file names."], ["keep picture file names", "Keep everything",
                                                    "Keep the exact text the user wants on a picture."])
    assert rules == ["Keep picture file names.", "Keep the exact text the user wants on a picture."]
    assert len(CA.merge([], [f"rule {i}" for i in range(40)])) == CA.MAX_RULES


if __name__ == "__main__":
    test_cost_counts_repeats_reasks_recalls(); test_acon_merge_dedupes_caps_and_refuses_keep_all(); print("2/2 ok")
