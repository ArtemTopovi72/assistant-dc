"""Tool argument models and JSON-schema generation.

Split out of tools.py: this half is a leaf — it holds no state, imports no
pipeline, and nothing above it in tools.py refers to these names. The handlers
never touch the args models; execute_tool does, through the ToolSpec registry.
Keeping them here means the schema contract can be read (and tested) without
pulling in image/search/deep_research.
"""
import re

import config
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# --- argument models ---------------------------------------------------------
#
# Single source of truth for tool arguments. Validators are deliberately
# forgiving where a small model is likely to misstep (out-of-range numbers are
# clamped, enum-ish values normalised, junk keys dropped) and strict only where
# guessing would be harmful (a missing query/description/region/instructions).

class _ToolArgs(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class SearchArgs(_ToolArgs):
    query: str = Field(..., min_length=1, description=(
        "A focused search query in keywords, not a full sentence. Include "
        "distinguishing details, and the year for recent events "
        "(e.g. 'Tokyo population 2024', not 'how many people live in Tokyo'). Ask the "
        "NEUTRAL question: never put in a name or figure you only guess as the "
        "answer — it steers the results toward your guess."))


class DeepResearchArgs(_ToolArgs):
    topic: str = Field(..., min_length=1, description=(
        "The research topic/question, as rich as possible "
        "(English preferred for source coverage). Ask the NEUTRAL question: never "
        "name a candidate answer you only guess — it steers the research to it."))
    depth: Literal["quick", "standard", "deep"] = Field("standard", description=(
        "How exhaustive to be. 'standard' is the default; 'deep' reads more pages "
        "and follows links; 'quick' is a fast pass."))

    @field_validator("depth", mode="before")
    @classmethod
    def _coerce_depth(cls, v):
        v = str(v or "").strip().lower()
        return v if v in ("quick", "standard", "deep") else "standard"


class GenerateImageArgs(_ToolArgs):
    description: str = Field(..., min_length=1, description=(
        "A clear English description of the subject and scene. The system "
        "expands it into a detailed prompt, so a concise sentence is enough."))

    @model_validator(mode="before")
    @classmethod
    def _remap_prompt(cls, values):
        # Gemma 4 uses 'prompt' instead of 'description' despite the schema.
        if isinstance(values, dict) and "prompt" in values and "description" not in values:
            values["description"] = values.pop("prompt")
        return values
    steps: Optional[int] = Field(None, ge=4, le=12, description=(
        "Optional sampling steps. Leave unset to use the default (8); "
        "only set if the user asks."))
    width: Optional[int] = Field(None, description=(
        "Optional width in pixels, multiple of 8. Leave UNSET unless this message names "
        "a size or a format - unset means the size they picked in Draw > Size. "
        "Stories/reels/vertical: 944x1680; YouTube cover/thumbnail/horizontal/wide: "
        "1680x944; square/post: 1264x1264."))
    height: Optional[int] = Field(None, description=(
        "Optional height in pixels, multiple of 8. See width."))
    seed: Optional[int] = Field(None, description=(
        "Optional random seed. Use -1 or leave unset for a random image."))

    @field_validator("steps", mode="before")
    @classmethod
    def _clamp_steps(cls, v):
        v = _as_int(v)
        return None if v is None else max(4, min(12, v))

    @field_validator("width", "height", mode="before")
    @classmethod
    def _round_to_8(cls, v):
        v = _as_int(v)
        if v is None or v <= 0:
            return None
        # Clamp to a sane range: absurd sizes would OOM/stall ComfyUI.
        return min(getattr(config, "IMAGE_MAX_SIDE", 2048),
                   max(getattr(config, "IMAGE_MIN_SIDE", 256), round(v / 8) * 8))

    @field_validator("seed", mode="before")
    @classmethod
    def _coerce_seed(cls, v):
        return _as_int(v)

    @model_validator(mode="after")
    def _orientation_follows_description(self):
        # "a wide landscape image for a YouTube banner" came with 1920x3508.
        w, h, d = self.width, self.height, self.description or ""
        # "квадратную" came as {"width": 1264} alone: the other side of the pair.
        pair = {1680: 944, 944: 1680, 1264: 1264}
        wide = re.search(r"\b(?:horizontal|landscape (?:format|orientation|image|shot)|widescreen|"
                         r"wide (?:format|image|banner|cover)|youtube|banner|16:9)\b", d, re.I)
        tall = re.search(r"\b(?:vertical|stor(?:y|ies)|reels?|tiktok|9:16)\b", d, re.I)
        # "a YouTube thumbnail ..." came with no size at all and rendered square.
        if not w and not h and bool(wide) != bool(tall):
            self.width, self.height = (1680, 944) if wide else (944, 1680)
            return self
        if bool(w) != bool(h):
            self.width, self.height = w or pair.get(h, h), h or pair.get(w, w)
            w, h = self.width, self.height
        if w and h and w != h:
            if (wide and not tall and h > w) or (tall and not wide and w > h):
                self.width, self.height = h, w
        return self


class GenerateVideoArgs(_ToolArgs):
    description: str = Field(..., min_length=1, description=(
        "What the video should show, in English. Describe the SHOT: what is in "
        "frame, what moves, what the camera does, and what it sounds like — H3 "
        "generates the audio too, so saying 'rain on a tin roof, no music' is "
        "meaningful. Spoken lines are the exception to English: quote them exactly "
        "as the user wrote them, in their language, also when redoing a clip — the "
        "clip speaks what is quoted. When references are attached, refer to them by their tags "
        "(<Picture 1>, <Video 1>) and say how they relate to the output, e.g. "
        "'<Picture 1> walks toward the camera' or 'the camera move from <Video 1> "
        "applied to <Picture 1>'."))
    seconds: Optional[float] = Field(None, ge=1.0, le=15.0, description=(
        "Clip length in seconds, 1-15. Leave unset unless the user names a length: then it "
        "follows the script (5-10s, enough for every action and spoken line in it). "
        "The real length snaps to the model's frame grid, so it may differ slightly."))
    aspect: Optional[Literal["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]] = Field(
        None, description=(
            "Shape of the clip. Leave unset to follow the source image, or 16:9 "
            "when there is no source. Use 9:16 when the user wants vertical: reels, "
            "TikTok, Shorts and stories are 9:16."))
    use_current_images: bool = Field(True, description=(
        "Use the picture(s) already in this conversation as input. TRUE (default) "
        "whenever the user says 'animate this', 'make a video from these' or similar. "
        "Set FALSE only for a video invented purely from text."))
    use_current_video: bool = Field(False, description=(
        "Also use the most recent VIDEO in the conversation as a reference — set TRUE "
        "when the user wants the motion/style of an existing clip carried over."))
    speakers: int = Field(0, ge=0, le=9, description=(
        "How many people TALK in the clip (0 = nobody speaks: scenery, music, an "
        "animal). With 1+ the user is asked once whether to use their own voice "
        "samples or the default voices."))
    use_my_voice: bool = Field(False, description=(
        "TRUE when the user wants the person in the clip to speak with THEIR OWN "
        "(cloned) voice — «моим голосом», «с моим голосом», «in my voice». The voice "
        "becomes <Audio 1>; write the spoken line in the description."))
    seed: Optional[int] = Field(None, description=(
        "Optional random seed. Leave unset for a new clip each time."))

    @model_validator(mode="before")
    @classmethod
    def _remap_prompt(cls, values):
        if isinstance(values, dict) and "prompt" in values and "description" not in values:
            values["description"] = values.pop("prompt")
        return values

    @field_validator("seconds", mode="before")
    @classmethod
    def _coerce_seconds(cls, v):
        if v in (None, "", "null"):
            return None
        try:
            return max(1.0, min(15.0, float(v)))
        except (TypeError, ValueError):
            return None

    @field_validator("seed", mode="before")
    @classmethod
    def _coerce_seed(cls, v):
        return _as_int(v)

    @field_validator("use_current_images", "use_current_video", "use_my_voice", mode="before")
    @classmethod
    def _coerce_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() not in ("false", "0", "no", "none", "")
        return bool(v) if v is not None else False


class RedrawImageArgs(_ToolArgs):
    mode: Optional[Literal["redraw"]] = Field(None, description=(
        "'redraw': re-render guided by the original's structure but follow new "
        "'instructions' — use when the user wants changes, or leave 'instructions' "
        "empty for a general quality pass. This is the only mode."))
    instructions: str = Field("", description=(
        "English description of the change to apply, e.g. 'make it snowy winter', "
        "'sharper face'. Leave empty for a general quality pass. "
        "Do NOT re-describe the whole scene or repeat the original subject — the current "
        "content carries over automatically, and re-describing it can revert earlier edits."))

    @field_validator("mode", mode="before")
    @classmethod
    def _coerce_mode(cls, v):
        v = str(v or "").strip().lower()
        return v if v == "redraw" else None


class InspectImageArgs(_ToolArgs):
    check: str = Field(..., min_length=1, description=(
        "What to verify, phrased as a concrete checklist/question. E.g. 'Is a "
        "fork held in his hand and pressing into the dough? Are the hand and fork "
        "free of artifacts (extra fingers, bent tines)?'"))
    earlier: int = Field(0, ge=0, le=3, description=(
        "How many pictures sent BEFORE the current one to look at as well (0 = only "
        "the current picture). Use 1 to compare two photos, 2 for three."))


class InpaintImageArgs(_ToolArgs):
    instructions: str = Field(..., min_length=1, description=(
        "What the edited region should look like, as a short English phrase "
        "describing the desired RESULT (not a command). E.g. 'wearing "
        "black-framed glasses', 'a blue dress shirt', 'a clean five-fingered hand'."))
    region: str = Field(..., min_length=1, description=(
        "The thing/area in the CURRENT image to change, named in plain words. The "
        "editor understands the whole image, so you can use specific, descriptive, "
        "or positional phrases — not just simple nouns. Good examples: 'glasses', "
        "'the mug', 'cat tail', 'puddle below the jar', 'left hand', 'blue shirt', "
        "'background behind the person'. It edits that thing and leaves the rest of "
        "the image unchanged."))


class TransferImageArgs(_ToolArgs):
    instructions: str = Field(..., min_length=1, description=(
        "What to transfer and where, in plain English referencing the images, e.g. "
        "'put the hat from the reference image onto the person', 'dress the person "
        "in the outfit from the other image', 'apply the hairstyle from image A'. "
        "The most recently loaded image is the target; earlier loaded images are the "
        "references."))
    roles: Optional[list] = Field(default=None, description=(
        "Optional role for each reference image, in load order (oldest first). One "
        "of: object_source, clothing_source, face_reference, hairstyle_reference, "
        "pose_reference, style_reference, identity_reference, scene_reference, "
        "lighting_reference. Omit to auto-infer from the instructions."))


class CalculateArgs(_ToolArgs):
    expression: str = Field(..., min_length=1, description=(
        "A Python math expression, e.g. '2**10', '(15*1.2)/3', "
        "'sqrt(144)', 'log(1000, 10)'"))


class ReadClipboardArgs(_ToolArgs):
    pass


class ForgetFactsArgs(_ToolArgs):
    what: str = Field("all", description=(
        "What to forget: 'all' to drop every saved fact about the user, or a few "
        "words naming the fact(s) to drop, e.g. 'аллергия', 'кот Барсик'."))


class LocalTimeArgs(_ToolArgs):
    city: str = Field(..., min_length=2, description="City, e.g. 'Токио' or 'New York'.")


class WeatherForecastArgs(_ToolArgs):
    city: str = Field(..., min_length=2, description="City name as the user wrote it, e.g. 'Екатеринбург'.")
    date: Optional[str] = Field(None, description="'YYYY-MM-DD' of the first day; omit for the next 24 hours.")
    days: int = Field(1, ge=1, le=7, description="How many days from `date` (2 for a weekend).")


class ExchangeRateArgs(_ToolArgs):
    codes: str = Field("USD", description=(
        "Currency codes, comma-separated, e.g. 'USD' or 'USD,EUR,CNY'."))


class SetReminderArgs(_ToolArgs):
    action: Literal["set", "list", "cancel"] = Field("set", description=(
        "'set' a new one, 'list' the pending ones, or 'cancel' the ones whose text "
        "matches `text` ('all' cancels every one)."))
    text: str = Field("", description=(
        "What to remind about, in the user's language, e.g. 'выпить воды'. For "
        "cancel: a word from the reminder, or 'all'."))
    minutes: Optional[float] = Field(None, description=(
        "In how many minutes from now ('через 2 минуты' -> 2, 'через час' -> 60)."))
    at: Optional[str] = Field(None, description=(
        "OR a clock time 'HH:MM' (today, or tomorrow if already past) or "
        "'YYYY-MM-DD HH:MM', local time."))
    repeat: Literal["none", "daily", "weekly"] = Field("none", description=(
        "'daily' for 'каждый день', 'weekly' for 'каждую неделю / по понедельникам'; "
        "'none' for a one-off."))


class RecallContextArgs(_ToolArgs):
    id: str = Field(..., min_length=2, max_length=16, description=(
        "The id from an [archived aXXXXXXX: ...] note in the conversation or in "
        "the working memory, e.g. 'a1b2c3d'."))


class FoldContextArgs(_ToolArgs):
    note: str = Field(..., min_length=2, max_length=200, description=(
        "One line naming the finished work to fold away, e.g. 'the logo drafts "
        "are done, the user chose v3'."))


class RememberFactArgs(_ToolArgs):
    fact: str = Field(..., min_length=1, max_length=500, description=(
        "One short, self-contained statement to remember permanently, in the "
        "user's language. Include WHO/WHAT it is about so it makes sense on its "
        "own later, e.g. 'Любимый цвет пользователя — изумрудный', "
        "'Кота пользователя зовут Барсик'. One fact per call."))


class FindPhotoArgs(_ToolArgs):
    query: str = Field(..., min_length=1, max_length=200, description=(
        "Who or what to find a real photo of, e.g. 'Sydney Sweeney portrait', "
        "'Eiffel Tower at night'. For a person, name them and optionally add 'portrait' "
        "or 'face'. English works best for image search."))
    require_face: bool = Field(True, description=(
        "Keep True (default) when finding a PERSON — only a photo containing a clear "
        "face is accepted. Set False only for non-person subjects (a building, a car)."))


class CreatePresentationArgs(_ToolArgs):
    topic: str = Field(..., min_length=1, max_length=500, description=(
        "What the presentation is about, in the user's own words, e.g. 'the history "
        "of the Trans-Siberian Railway for a school class'."))
    slides: int = Field(0, ge=0, le=30, description=(
        "How many slides, if the user named a number. 0 lets the planner decide "
        "(5-12)."))
    edit_previous: bool = Field(False, description=(
        "True when the user wants to CHANGE the deck just made (add, remove, "
        "rename or rewrite a slide) rather than a new one; `topic` then holds "
        "the change itself, e.g. 'add a slide about the moons of Mars'."))
    illustrate: bool = Field(True, description=(
        "Whether to illustrate slides with photos found on the web. True by "
        "default; set False if the user wants a text-only deck."))


class FixHandsArgs(_ToolArgs):
    # No arguments: FireRed is the only edit engine. The 'engine' switch to
    # base Qwen-Image-Edit was removed with that model.
    pass


class FixArtifactArgs(_ToolArgs):
    region: str = Field(..., min_length=1, description=(
        "A short noun phrase naming WHERE the flaw is, so it can be located in the image, "
        "e.g. 'the seam on the left shoulder', 'the smear above the table', 'the background "
        "near her elbow', 'the blurry patch on the wall'. Be specific about location."))
    issue: str = Field("", description=(
        "What is wrong / what it should look like, e.g. 'smooth this harsh transition', "
        "'remove this smear', 'fix this melted edge'. Leave empty for a generic seamless "
        "clean-up of the region."))


def _as_int(v) -> Optional[int]:
    """Best-effort int coercion for LLM-supplied numbers ('8', 8.0, '');
    returns None for anything unusable so the pipeline default applies."""
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


# --- schema generation -------------------------------------------------------

def _clean_schema(node):
    """Normalise pydantic's model_json_schema output into the plain JSON-schema
    shape LLM tool APIs expect: drop 'title' noise and collapse Optional fields'
    anyOf [X, null] wrapper down to X."""
    if isinstance(node, dict):
        node = {k: v for k, v in node.items() if k != "title"}
        variants = [v for v in node.get("anyOf", []) if v.get("type") != "null"]
        if len(variants) == 1:
            merged = {**variants[0], **{k: v for k, v in node.items() if k != "anyOf"}}
            return _clean_schema(merged)
        return {k: _clean_schema(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_clean_schema(v) for v in node]
    return node


def _fn_schema(name: str, description: str, model: type) -> dict:
    """Build the OpenAI-style function schema for a tool from its args model."""
    params = _clean_schema(model.model_json_schema())
    params.setdefault("properties", {})
    params.setdefault("required", [])
    return {"type": "function",
            "function": {"name": name, "description": description, "parameters": params}}

# --- coding sandbox ---------------------------------------------------------
# Paths are always RELATIVE to the user's sandbox root. The description says so
# in every field: the model is otherwise happy to send C:/Users/... and then be
# told off by the containment check, which wastes a round each time.


class ListFilesArgs(_ToolArgs):
    path: str = Field(".", description=(
        "Folder to list, relative to the project root. '.' for the top level."))
    depth: int = Field(3, ge=1, le=8, description=(
        "How many folder levels to show at once. 3 covers a mod's "
        "data/<ns>/tags; raise it rather than listing folder by folder."))


class ReadFileArgs(_ToolArgs):
    path: str = Field(..., min_length=1, description=(
        "File to read, relative to the project root, e.g. "
        "'data/thief/tags/block/medium.json'."))
    start: int = Field(1, ge=1, description=(
        "First line to show (1-based). A long file comes back in windows; the "
        "result says which start to pass for the next one."))
    lines: int = Field(120, ge=20, le=600, description="How many lines to show at most.")


class WriteFileArgs(_ToolArgs):
    # Code is indentation: stripping `content`/`old`/`new` turned
    # "        return x" into "return x" -- every indented edit became an
    # IndentationError and was reverted by the lint gate (found 2026-09-24).
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=False)

    @field_validator("path", mode="before")
    @classmethod
    def _strip_path(cls, v):
        return v.strip() if isinstance(v, str) else v
    path: str = Field(..., min_length=1, description=(
        "File to create or overwrite, relative to the project root. "
        "Parent folders are created for you."))
    content: str = Field("", description=(
        "The contents: the whole NEW file, or with append=true the part to add at "
        "its end. At most 150 lines per call."))
    append: bool = Field(False, description=(
        "true = add `content` to the END of an existing file (how a long file is "
        "written: first part, then append the rest part by part)."))


class EditFileArgs(_ToolArgs):
    # Code is indentation: stripping `content`/`old`/`new` turned
    # "        return x" into "return x" -- every indented edit became an
    # IndentationError and was reverted by the lint gate (found 2026-09-24).
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=False)

    @field_validator("path", mode="before")
    @classmethod
    def _strip_path(cls, v):
        return v.strip() if isinstance(v, str) else v
    path: str = Field(..., min_length=1, description=(
        "File to edit, relative to the project root."))
    old: str = Field("", description=(
        "The exact text to replace, copied from the file including indentation. "
        "It must appear EXACTLY ONCE; include surrounding lines to make it "
        "unique. Leave empty when you give start_line/end_line instead."))
    new: str = Field("", description="What to put in its place. Empty deletes it.")
    start_line: int = Field(0, ge=0, description=(
        "Instead of `old`: first line to replace, as numbered by read_file (1-based)."))
    end_line: int = Field(0, ge=0, description=(
        "Instead of `old`: last line to replace (inclusive). start_line=end_line "
        "replaces one line."))


class RagAddArgs(_ToolArgs):
    path: str = Field(..., min_length=1, description=(
        "An existing file in the working folder to add to this chat's "
        "knowledge base, e.g. 'report.pdf' or 'notes.txt'. Use list_files "
        "first if you are not sure of the exact name."))


class RagSearchArgs(_ToolArgs):
    query: str = Field(..., min_length=1, description=(
        "What to look up in this chat's knowledge base, in plain words or "
        "keywords."))
    k: int = Field(5, ge=1, le=20, description="How many passages to return.")


class SearchFilesArgs(_ToolArgs):
    pattern: str = Field(..., min_length=1, description=(
        "Regular expression to look for, e.g. 'villager_job_sites'."))
    path: str = Field(".", description="Folder to search in, relative to the root.")
    offset: int = Field(0, ge=0, description=(
        "Skip this many matching FILES. The result says how many matched and "
        "how many are shown; pass the offset it suggests to see the next "
        "page, with the same pattern."))


class FindContentArgs(_ToolArgs):
    query: str = Field(..., min_length=1, description=(
        "What to find, in plain words: 'a bridge', 'a red car', 'a text with "
        "the word Konjic', 'invoice number 4471'. Pictures are searched by what "
        "they show, text files by their contents."))
    path: str = Field(".", description="Folder to search, relative to the root.")
    kind: str = Field("any", description=(
        "'photos' to look only at pictures, 'text' only at text files, 'any' "
        "(default) for both."))


class DedupePhotosArgs(_ToolArgs):
    paths: list[str] = Field(default_factory=list, description=(
        "The photos to de-duplicate, relative paths -- normally the ones "
        "find_content returned. Leave empty to reuse exactly those."))
    path: str = Field("", description=(
        "Folder whose EVERY picture to de-duplicate. Only when the user asks "
        "to clean the whole folder, never to pick photos for a themed collage."))
    whole_folder: bool = Field(False, description=(
        "true only when the user wants EVERY picture in `path` compared; "
        "otherwise the photos find_content found are used."))
    keep_dir: str = Field("", description=(
        "Optional folder to COPY the kept photos into, e.g. 'unique'. The "
        "originals are never deleted."))


class DeletePathArgs(_ToolArgs):
    path: str = Field(..., min_length=1, description=(
        "File or folder to delete, relative to the working folder. A folder "
        "is removed with everything inside it."))


class UnpackArchiveArgs(_ToolArgs):
    path: str = Field(..., min_length=1, description=(
        "The .zip or .jar to extract, relative to the project root."))
    dest: str = Field("", description=(
        "Folder to extract into. Leave empty for '<name>_unpacked'."))


class PackArchiveArgs(_ToolArgs):
    path: str = Field(..., min_length=1, description=(
        "Folder to zip up, relative to the project root."))
    output: str = Field("result.zip", description=(
        "Name of the archive to produce. This file is what the user receives."))


class CodeOutlineArgs(_ToolArgs):
    path: str = Field(".", description="Folder or file to outline, relative to the root.")


class RunTestsArgs(_ToolArgs):
    path: str = Field(".", description="Test file or folder, relative to the root.")
    filter: str = Field("", description="Optional pytest -k expression to run only matching tests.")
    timeout: Optional[int] = Field(None, ge=5, le=1800, description="Seconds before the run is stopped.")


class PlanStep(_ToolArgs):
    step: str = Field(..., min_length=1, description="What this step does.")
    status: Literal["todo", "doing", "done"] = Field("todo", description="todo, doing or done.")


class UpdatePlanArgs(_ToolArgs):
    steps: list[PlanStep] = Field(default_factory=list, description=(
        "The WHOLE plan every time (not a diff), in order. Empty list clears it."))


class UndoEditArgs(_ToolArgs):
    path: str = Field(..., min_length=1, description="The file whose last write_file/edit_file to take back.")


class OzonSearchArgs(_ToolArgs):
    query: str = Field(..., min_length=1, description=(
        "What to look for on Ozon, in RUSSIAN as a shopper would type it, e.g. "
        "'беспроводные наушники с шумоподавлением'."))
    sort: Literal["popular", "price", "price_desc", "rating", "new", "discount"] = Field(
        "popular", description="price = cheapest first; rating = best rated first.")
    price_min: Optional[int] = Field(None, ge=0, description="Lowest price in rubles.")
    price_max: Optional[int] = Field(None, ge=0, description="Highest price in rubles.")
    limit: int = Field(10, ge=1, le=20, description="How many products to return.")
    max_delivery_days: Optional[int] = Field(None, ge=0, le=60, description=(
        "Keep only products Ozon delivers within this many days (0 = today, 1 = tomorrow)."))


class OzonProductArgs(_ToolArgs):
    product: str = Field(..., min_length=1, description=(
        "An Ozon product link, SKU number, or slug from ozon_search results."))
    look: bool = Field(False, description=(
        "Also LOOK at the product photos with vision (size, build, what is in the "
        "box, visible defects, whether it matches the title)."))
    question: Optional[str] = Field(None, description=(
        "What to check on the photos, e.g. 'is the kettle glass or plastic?'."))


class OzonShopArgs(_ToolArgs):
    request: str = Field(..., min_length=2, description=(
        "What the user wants, in their words with every detail: 'самый дешёвый нормальный "
        "самогонный аппарат 20 л', or a set/occasion/list: 'всё для пикника на 4 человек до "
        "3000 ₽', 'купить: скатерть, 8 тарелок, приборы'."))
    strategy: Literal["balanced", "cheap", "best", "fast"] = Field(
        "balanced", description="cheap = lowest price that is still fine; best = best "
        "reviewed; fast = earliest delivery; balanced = value for money.")
    clarified: bool = Field(False, description=(
        "true once the user answered the clarifying questions a previous ozon_shop call "
        "returned, or when the request already came with those details."))
    add_to_cart: bool = Field(False, description=(
        "Put the chosen products into the user's shopping list (for 'собери корзину')."))
    by_photo: bool = Field(False, description=(
        "true when the user wants the product IN THE PICTURE (they sent or pointed at a "
        "photo: 'найди такое же', 'где купить это'): Ozon's photo search finds it; the "
        "request then holds only their wishes and priority (or is 'как на фото')."))

    @model_validator(mode="before")
    @classmethod
    def _query_is_request(cls, v):
        # The model reuses ozon_search's shape ({query, price_max}); rejecting it
        # dropped the turn to a bare search that read no reviews (live 2026-09-28,
        # "наушники до 3000, чтоб без брака").
        if not isinstance(v, dict):
            return v
        v = dict(v)
        if not v.get("request") and v.get("query"):
            v["request"] = str(v["query"])
        cap = v.get("price_max")
        if cap and v.get("request") and str(cap) not in str(v["request"]):
            # a budget passed beside the request was ignored: a 3000 ₽ ask came
            # back as the 383 ₽ cheapest pair
            v["request"] = f'{v["request"]} до {cap} ₽'
        return v


class OzonSetLocationArgs(_ToolArgs):
    place: str = Field("", description=(
        "City or street address in Russia, e.g. 'Казань' or 'Москва, Тверская 7'. "
        "The nearest Ozon pickup point becomes the delivery location. Empty = "
        "just report the current one."))


class OzonCartArgs(_ToolArgs):
    action: Literal["show", "add", "remove", "clear"] = Field(
        "show", description="show the list with the total, add/remove one product, or clear it.")
    product: Optional[str] = Field(None, description="Ozon link or SKU (for add/remove).")
    qty: int = Field(1, ge=1, le=99, description="How many to add.")


class OzonReviewsArgs(_ToolArgs):
    product: str = Field(..., min_length=1, description=(
        "An Ozon product link, SKU number, or slug."))
    limit: int = Field(15, ge=1, le=30, description="How many recent reviews to read.")


class RunCodeArgs(_ToolArgs):
    code: str = Field(..., min_length=1, description=(
        "A complete Python script. It runs with the project root as its working "
        "directory, so open files by their relative path. print() what you need "
        "to see -- that output is all you get back. Photos: `import "
        "assistant_tools` -- dedupe(paths) keeps the sharpest of every "
        "same-scene group (re-shots, bursts), make_collage(paths, out), "
        "sharpness(path), list_images(folder). Never hand-roll an image hash: "
        "it misses re-shots and the user sees the copies you kept."))
    timeout: int = Field(60, description=(
        "Seconds to allow before the script is killed. Max 300."))


class InstallPackagesArgs(_ToolArgs):
    packages: list = Field(..., description=(
        "Package names to install, e.g. ['pillow']. Names only -- no flags, no "
        "URLs, no version pins with options."))
