"""Containment for the coding agent's filesystem access.

Every path here is model-generated text, not a user's click, so the escapes
below are the ordinary case rather than an attack scenario: a model that guesses
`../../config.json` is not malicious, it is wrong, and the difference does not
matter to the filesystem.

Windows offers more ways out than POSIX, and the drive-relative form is the
quiet one: `C:notes.txt` looks relative, is not, and resolves against whatever
C:'s current directory happens to be.
"""
import sys, os, json, zipfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from pathlib import Path
from code_sandbox import (Sandbox, SandboxError, sandbox_for, sandbox_key,
                          AGENT_DIR)


@pytest.fixture
def box(tmp_path):
    s = Sandbox(tmp_path / "proj")
    (s.root / "config").mkdir()
    (s.root / "config" / "settings.json").write_text('{"a": 1}', encoding="utf-8")
    return s


# --- containment ------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "../outside.txt", "../../outside.txt", "config/../../outside.txt",
    "C:/Windows/System32/drivers/etc/hosts", "C:\\Windows\\win.ini",
    "/etc/passwd", "//server/share/x.txt", "//./PIPE/x",
    "C:notes.txt", "", "   ",
])
def test_escapes_are_refused(box, bad):
    with pytest.raises(SandboxError):
        box.resolve(bad)


def test_ordinary_paths_resolve(box):
    assert box.resolve("config/settings.json").exists()
    assert box.resolve("new/deep/file.txt").parent.name == "deep"
    # a `..` that stays inside is fine — only LEAVING is refused
    assert box.resolve("config/../config/settings.json").exists()


def test_a_symlinked_directory_cannot_smuggle_a_path_out(box, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s", encoding="utf-8")
    try:
        (box.root / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need privileges on this machine")
    with pytest.raises(SandboxError):
        box.resolve("link/secret.txt")


# --- reading and writing ----------------------------------------------------

def test_read_write_edit(box):
    assert box.read_text("config/settings.json") == '{"a": 1}'
    box.write_text("notes/todo.md", "hello")
    assert box.read_text("notes/todo.md") == "hello"
    box.replace_once("notes/todo.md", "hello", "bye")
    assert box.read_text("notes/todo.md") == "bye"


def test_an_ambiguous_edit_is_refused(box):
    """Two matches means the model has not identified the site it thinks it
    has; replacing the first would edit the wrong line and report success."""
    box.write_text("a.txt", "x = 1\nx = 1\n")
    with pytest.raises(SandboxError, match="appears 2 times"):
        box.replace_once("a.txt", "x = 1", "x = 2")


def test_a_missing_edit_target_is_refused(box):
    with pytest.raises(SandboxError, match="not in"):
        box.replace_once("config/settings.json", "nope", "x")


def test_binary_files_are_not_read_as_text(box):
    (box.root / "art.png").write_bytes(b"\x89PNG\x00\x00binary")
    with pytest.raises(SandboxError, match="not a text file"):
        box.read_text("art.png")


def test_a_log_is_text(box):
    # live 10-03: crash.log / assistant_app.log refused as «not a text file»
    (box.root / "crash.log").write_text("Traceback: boom", encoding="utf-8")
    assert "boom" in box.read_text("crash.log")


def test_a_huge_file_is_refused_with_advice(box):
    (box.root / "big.json").write_bytes(b"{}" * 200_000)
    with pytest.raises(SandboxError, match="Search it"):
        box.read_text("big.json")


# --- search -----------------------------------------------------------------

def test_search_finds_the_line(box):
    box.write_text("data/thief/tags.json", '{"values": ["#c:villager_job_sites"]}')
    assert any("tags.json" in h for h in box.search("villager_job_sites"))


def test_a_bad_pattern_is_a_message_not_a_crash(box):
    with pytest.raises(SandboxError, match="Bad search pattern"):
        box.search("(unclosed")


# --- archives ---------------------------------------------------------------

def _zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def test_unpack_and_pack_round_trip(box):
    _zip(box.root / "mod.jar",
         {"data/thief/tags/block/medium.json": '{"values": []}',
          "META-INF/MANIFEST.MF": "Manifest-Version: 1.0"})
    assert "2 files" in box.unpack("mod.jar")
    assert box.read_text("mod_unpacked/data/thief/tags/block/medium.json")
    packed = box.pack("mod_unpacked", "fixed.zip")
    assert packed.exists() and zipfile.is_zipfile(packed)


def test_zip_slip_is_refused(box):
    """An entry named ../../evil.txt is an ordinary thing to find in a hostile
    archive, and the stdlib extraction helpers have honoured it before."""
    _zip(box.root / "evil.zip", {"../../../evil.txt": "pwned"})
    with pytest.raises(SandboxError):
        box.unpack("evil.zip")
    assert not (box.root.parent.parent / "evil.txt").exists()


def test_a_non_archive_is_refused(box):
    box.write_text("notes.txt", "not a zip")
    with pytest.raises(SandboxError, match="not an archive"):
        box.unpack("notes.txt")


# --- per-user isolation -----------------------------------------------------

def test_each_user_gets_their_own_root(tmp_path):
    a = sandbox_for(12345, base=tmp_path)
    b = sandbox_for(67890, base=tmp_path)
    assert a.root != b.root
    a.write_text("config.json", "A")
    b.write_text("config.json", "B")
    assert a.read_text("config.json") == "A"
    assert b.read_text("config.json") == "B"


def test_one_user_cannot_reach_another(tmp_path):
    a = sandbox_for("alice", base=tmp_path)
    sandbox_for("bob", base=tmp_path).write_text("secret.txt", "s")
    with pytest.raises(SandboxError):
        a.resolve("../bob/secret.txt")


@pytest.mark.parametrize("hostile", ["../../etc", "a/b", "..", "C:\\x", ""])
def test_a_hostile_id_cannot_escape_the_base(tmp_path, hostile):
    box = sandbox_for(hostile, base=tmp_path)
    assert Path(tmp_path).resolve() in box.root.parents, box.root


def test_the_key_is_stable(tmp_path):
    assert sandbox_key(12345) == "12345"
    assert sandbox_for(12345, base=tmp_path).root == sandbox_for(12345, base=tmp_path).root


# --- reset to factory -------------------------------------------------------

def test_reset_empties_the_sandbox_but_keeps_it_usable(box):
    box.write_text("a/b/c.txt", "x")
    box.write_text("top.txt", "y")
    assert box.reset() >= 2
    assert box.root.exists() and not any(box.root.iterdir())
    box.write_text("fresh.txt", "z")
    assert box.read_text("fresh.txt") == "z"


def test_reset_touches_only_this_user(tmp_path):
    a = sandbox_for("alice", base=tmp_path)
    b = sandbox_for("bob", base=tmp_path)
    a.write_text("x.txt", "A")
    b.write_text("x.txt", "B")
    a.reset()
    assert b.read_text("x.txt") == "B", "reset crossed into another sandbox"


def test_reset_does_not_follow_a_symlink_out(box, tmp_path):
    outside = tmp_path / "keep"
    outside.mkdir()
    (outside / "precious.txt").write_text("keep me", encoding="utf-8")
    try:
        (box.root / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need privileges on this machine")
    box.reset()
    assert (outside / "precious.txt").exists(), "reset deleted through a symlink"


def test_usage_reports_what_is_held(box):
    box.write_text("a.txt", "12345")
    files, total = box.usage()
    assert files >= 1 and total >= 5


# --- errors that carry their own correction ---------------------------------

def test_a_missing_file_error_lists_what_is_there(box):
    """Measured end to end: asked about "the Thief jar", the model guessed
    'Thief.jar', got "does not exist", and told the user to upload a file that
    was already in the folder as thief-1.21.1.jar. The error now carries the
    answer, which turns a dead end into one more round."""
    box.write_text("thief-1.21.1.jar", "x")
    with pytest.raises(SandboxError) as exc:
        box.read_text("Thief.jar")
    assert "thief-1.21.1.jar" in str(exc.value), str(exc.value)


def test_the_listing_names_the_folder_it_describes(box):
    box.write_text("sub/deep.json", "{}")
    with pytest.raises(SandboxError) as exc:
        box.read_text("sub/other.json")
    assert "sub contains" in str(exc.value) and "deep.json" in str(exc.value)


def test_unpacking_a_misspelled_archive_says_what_is_available(box):
    box.write_text("morevillagers-6.0.0.jar", "x")
    with pytest.raises(SandboxError) as exc:
        box.unpack("morevillagers.jar")
    assert "morevillagers-6.0.0.jar" in str(exc.value)


def test_an_empty_folder_says_so_rather_than_listing_nothing(box, tmp_path):
    empty = Sandbox(tmp_path / "empty")
    with pytest.raises(SandboxError) as exc:
        empty.read_text("anything.txt")
    assert "empty" in str(exc.value).lower()


def test_search_matches_the_path_as_well_as_the_content(box):
    """Measured end to end: asked to edit the tag "break_protected/medium", the
    model searched for exactly that and found nothing three times over — the
    string lives in the file's LOCATION, not in any line of it, which is how
    data packs and most convention-based layouts are organised."""
    box.write_text("data/thief/tags/block/break_protected/medium.json",
                   '{"values": ["#c:chests"]}')
    hits = box.search("break_protected/medium")
    assert hits, "a path-only match was not found"
    assert any("filename match" in h for h in hits), hits


def test_a_path_match_is_labelled_as_one(box):
    """So the model does not mistake a filename for a line of the file."""
    box.write_text("config/volume.json", '{"a": 1}')
    hits = box.search("volume")
    assert any(h.endswith("(filename match)") for h in hits), hits


def test_content_matches_still_work(box):
    box.write_text("a.json", 'x\n{"values": ["#c:villager_job_sites"]}\n')
    hits = box.search("villager_job_sites")
    assert any(":2:" in h for h in hits), hits


def test_a_failed_edit_shows_the_real_lines(box):
    """An exact-match edit usually fails on a detail the model cannot see from
    memory — two spaces instead of four, a trailing comma, a quote style.
    Showing the actual line fixes it in one step; "copy it exactly" does not."""
    box.write_text("m.json",
                   '{\n    "values": [\n        "#c:chests",\n'
                   '        "#c:villager_job_sites"\n    ]\n}')
    # A genuine miss: the model dropped the leading '#', which is the kind of
    # detail it cannot check from memory. A merely shorter indent would still
    # match, since the comparison is on substrings, not whole lines.
    with pytest.raises(SandboxError) as exc:
        box.replace_once("m.json", '"c:villager_job_sites"', '"c:x"')
    msg = str(exc.value)
    assert "closest lines" in msg, msg
    assert "villager_job_sites" in msg, msg


def test_no_close_line_means_no_misleading_suggestion(box):
    box.write_text("m.json", '{"a": 1}')
    with pytest.raises(SandboxError) as exc:
        box.replace_once("m.json", "totally unrelated content here", "x")
    assert "closest lines" not in str(exc.value)


# --- a real modpack shape ------------------------------------------------------

def _jar_bytes(tag):
    import io as _io, zipfile as _zf, json as _json
    buf = _io.BytesIO()
    with _zf.ZipFile(buf, "w") as z:
        z.writestr(f"data/{tag}/tag.json", _json.dumps({"mod": tag}))
    return buf.getvalue()


def test_two_mods_with_the_same_filename_do_not_merge(tmp_path):
    """A modpack is mods/*.jar, and jar names repeat across folders.

    Unpacking to the ROOT put both "thief.jar"s into one thief_unpacked, the
    second silently overwriting into the first -- so the model would read one
    mod's tag file believing it was the other's, and answer confidently about a
    file that never existed in that shape.
    """
    import zipfile
    box = Sandbox(tmp_path / "sbx")
    with zipfile.ZipFile(box.root / "pack.zip", "w") as z:
        z.writestr("mods/thief.jar", _jar_bytes("a"))
        z.writestr("extra/thief.jar", _jar_bytes("b"))

    box.unpack("pack.zip")
    box.unpack("pack_unpacked/mods/thief.jar")
    box.unpack("pack_unpacked/extra/thief.jar")

    assert json.loads(box.read_text(
        "pack_unpacked/mods/thief_unpacked/data/a/tag.json"))["mod"] == "a"
    assert json.loads(box.read_text(
        "pack_unpacked/extra/thief_unpacked/data/b/tag.json"))["mod"] == "b"


def test_an_archive_at_the_root_still_unpacks_at_the_root(tmp_path):
    import zipfile
    box = Sandbox(tmp_path / "sbx")
    with zipfile.ZipFile(box.root / "thief.jar", "w") as z:
        z.writestr("pack.mcmeta", "{}")
    out = box.unpack("thief.jar")
    assert out.startswith("thief_unpacked"), out
    assert box.read_text("thief_unpacked/pack.mcmeta") == "{}"


def test_an_explicit_destination_is_still_honoured(tmp_path):
    import zipfile
    box = Sandbox(tmp_path / "sbx")
    with zipfile.ZipFile(box.root / "a/deep/thief.jar".replace("/", os.sep)
                         if False else box.root / "thief.jar", "w") as z:
        z.writestr("x.txt", "hi")
    box.unpack("thief.jar", "somewhere/else")
    assert box.read_text("somewhere/else/x.txt") == "hi"


def test_the_delivered_archive_carries_no_working_debris(tmp_path):
    """The full modpack round trip: edit a file inside a jar inside a pack,
    repack both, and check what the user actually receives.

    Measured: the pack that went back carried mods/thief_unpacked/ -- a second,
    extracted copy of a mod that had already been repacked into its jar beside
    it. The archive was correct AND obviously wrong, at double the size.
    """
    import io as _io, zipfile
    box = Sandbox(tmp_path / "sbx")
    inner = _io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("data/thief/tags/block/medium.json",
                   json.dumps({"values": ["#c:chests"]}))
    with zipfile.ZipFile(box.root / "pack.zip", "w") as z:
        z.writestr("mods/thief.jar", inner.getvalue())
        z.writestr("config/thief.toml", "protect = true\n")

    box.unpack("pack.zip")
    box.unpack("pack_unpacked/mods/thief.jar")
    box.replace_once("pack_unpacked/mods/thief_unpacked/data/thief/tags/block/medium.json",
                     '"#c:chests"', '"#c:chests", "morevillagers:trading_table"')
    box.pack("pack_unpacked/mods/thief_unpacked", "pack_unpacked/mods/thief.jar")
    out = box.pack("pack_unpacked", "pack_fixed.zip")

    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        jar = z.read("mods/thief.jar")
    assert sorted(names) == ["config/thief.toml", "mods/thief.jar"], names

    with zipfile.ZipFile(_io.BytesIO(jar)) as z:
        data = json.loads(z.read("data/thief/tags/block/medium.json"))
    assert "morevillagers:trading_table" in data["values"], data


def test_packing_an_unpacked_folder_itself_is_the_normal_case(tmp_path):
    """Only NESTED debris is skipped -- "pack thief_unpacked back into a jar" is
    the ordinary flow and must still produce a full archive."""
    import zipfile
    box = Sandbox(tmp_path / "sbx")
    with zipfile.ZipFile(box.root / "thief.jar", "w") as z:
        z.writestr("pack.mcmeta", "{}")
        z.writestr("data/x.json", "{}")
    box.unpack("thief.jar")
    out = box.pack("thief_unpacked", "thief_fixed.jar")
    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == ["data/x.json", "pack.mcmeta"], z.namelist()


def test_the_private_agent_folder_never_ships(tmp_path):
    import zipfile
    box = Sandbox(tmp_path / "sbx")
    box.write_text("work/keep.txt", "keep me")
    box.write_text(f"work/{AGENT_DIR}/site-packages/junk.py", "print(1)")
    out = box.pack("work", "out.zip")
    with zipfile.ZipFile(out) as z:
        assert z.namelist() == ["keep.txt"], z.namelist()


def test_the_private_agent_folder_is_not_listed(tmp_path):
    box = Sandbox(tmp_path / "sbx")
    box.write_text("notes.txt", "hi")
    box.write_text(f"{AGENT_DIR}/site-packages/x.py", "print(1)")
    listing = box.list_dir(".")
    assert any(e.startswith("notes.txt") for e in listing), listing
    assert not any(AGENT_DIR in e for e in listing), listing


def test_but_it_is_still_reachable_when_named(tmp_path):
    """Hidden from the listing, not walled off: run_code writes its scripts
    there and has to be able to read them back."""
    box = Sandbox(tmp_path / "sbx")
    box.write_text(f"{AGENT_DIR}/run_1.py", "print(1)")
    assert box.read_text(f"{AGENT_DIR}/run_1.py") == "print(1)"


# --- walks must not follow links out of the tree -----------------------------
# run_code's container writes into the sandbox and can leave a symlink to an
# absolute path; the HOST then follows it. resolve() guards named paths, but
# search() and pack() walk the tree and used to read/zip whatever a link hit.

def _link_or_skip(link: Path, target: Path):
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available here")


def test_search_does_not_follow_a_link_out(box, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOKEN=hunter2", encoding="utf-8")
    _link_or_skip(box.root / "leak.txt", secret)
    hits = box.search("hunter2")
    assert not any("hunter2" in h for h in hits), hits


def test_pack_does_not_zip_a_file_outside(box, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOKEN=hunter2", encoding="utf-8")
    _link_or_skip(box.root / "config" / "leak.txt", secret)
    out = box.pack("config", "out.zip")
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "leak.txt" not in names, names
        assert "settings.json" in names


def test_list_does_not_descend_a_linked_dir(box, tmp_path):
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (outside / "private.txt").write_text("x", encoding="utf-8")
    _link_or_skip(box.root / "linked", outside)
    listing = box.list_dir(".", depth=3)
    assert not any("private.txt" in e for e in listing), listing


# --- unpack size gate ---------------------------------------------------------

def test_unpack_refuses_a_decompression_bomb(box, monkeypatch):
    import code_sandbox
    monkeypatch.setattr(code_sandbox, "MAX_UNPACK_BYTES", 100_000)
    zp = box.root / "bomb.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("zeros.bin", b"\0" * 1_000_000)     # ~1 KB compressed
    with pytest.raises(SandboxError, match="limit"):
        box.unpack("bomb.zip")
    left = box.root / ("bomb" + code_sandbox.UNPACK_SUFFIX) / "zeros.bin"
    assert not left.exists() or left.stat().st_size <= 100_000


def test_copy_is_counted_whatever_the_header_claims(monkeypatch):
    """The declared file_size is the archive's own claim; the copy is counted."""
    import io
    import code_sandbox
    monkeypatch.setattr(code_sandbox, "MAX_UNPACK_BYTES", 3 * 1024 * 1024)
    budget = [code_sandbox.MAX_UNPACK_BYTES]
    with pytest.raises(SandboxError, match="limit"):
        Sandbox._copy_capped(io.BytesIO(b"\0" * (5 * 1024 * 1024)), io.BytesIO(),
                             budget, "x.zip")


def test_unpack_normal_archive_still_works(box):
    zp = box.root / "mod.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("a/b.txt", "hello")
        zf.writestr("c.json", "{}")
    out = box.unpack("mod.zip")
    assert "(2 files)" in out
    assert (box.root / "mod_unpacked" / "a" / "b.txt").read_text() == "hello"
