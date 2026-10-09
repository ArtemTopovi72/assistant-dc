---
type: Domain
description: Fictional persona scripts and cast-preset JSON for the Madhouse room and the single-assistant personality override.
tags: [personas, madhouse, roleplay]
---

# personas

## What it does

[`personalities/`](../../personalities) holds fictional, scripted character write-ups and ready-made casts built from them. Two unrelated features load from the same directory: the Madhouse multi-character chat room, which assigns one persona and one voice clip to each seat in a scene, and the single-assistant Settings dialog, which lets the user swap the one assistant's personality for one of these `.txt` files. Every character here is an invented role for a genre scene (interrogation room, spaceship crew, garage, theater); none names or voices a real person.

## How it works

Each `.txt` file is a short, second-person character brief: who the character is in the scene, how they talk (speech tics, pacing, favorite words), and the flaw or tension that drives them in conversation. A `cast_*.json` file is a JSON array of `{id, name, voice, personality, prompt}` entries, where `voice` names a `*_short.wav` reference clip at the repo root and `personality` names one of these `.txt` files.

Two ways to attach a persona to a cast entry:

- **File-backed.** `personality` points at a `.txt` file and `prompt` is empty; [`gui/gui_madhouse_cast.py`](../../gui/gui_madhouse_cast.py)'s `_read_personality` reads that file at load time. All four scenario presets below use this form.
- **Inline.** `personality` is empty and `prompt` carries the character text directly in the JSON. [`personalities/madhouse_cast.json`](../../personalities/madhouse_cast.json) uses this for its DC, Лёха, and Сан entries: none of them has a matching `.txt` file.

When a saved cast is reloaded, an explicit `prompt` wins over re-reading the `personality` file ([`gui/gui_madhouse_cast.py`](../../gui/gui_madhouse_cast.py)'s `_load_characters`), so an inline-authored entry keeps its exact wording even if a same-named `.txt` is edited afterward, while a file-backed entry always re-reads the current file contents. A missing or empty `personality` falls back to a generic one-line placeholder describing the character as an unnamed room participant, rather than failing to load the cast.

The four scenario presets, and the mixed example cast that exercises both authoring modes:

| Cast file | Scenario | Character | Role | Persona file |
|---|---|---|---|---|
| [`personalities/cast_dopros.json`](../../personalities/cast_dopros.json) | interrogation room | Веретенников | investigator | [`personalities/dopros_sledovatel.txt`](../../personalities/dopros_sledovatel.txt) |
| [`personalities/cast_dopros.json`](../../personalities/cast_dopros.json) | interrogation room | Зинаида | witness | [`personalities/dopros_svidetel.txt`](../../personalities/dopros_svidetel.txt) |
| [`personalities/cast_dopros.json`](../../personalities/cast_dopros.json) | interrogation room | Марк | suspect | [`personalities/dopros_podozrevaemyi.txt`](../../personalities/dopros_podozrevaemyi.txt) |
| [`personalities/cast_ekipazh.json`](../../personalities/cast_ekipazh.json) | spaceship crew | Дорн | captain | [`personalities/ekipazh_kapitan.txt`](../../personalities/ekipazh_kapitan.txt) |
| [`personalities/cast_ekipazh.json`](../../personalities/cast_ekipazh.json) | spaceship crew | Марта | engineer | [`personalities/ekipazh_inzhener.txt`](../../personalities/ekipazh_inzhener.txt) |
| [`personalities/cast_ekipazh.json`](../../personalities/cast_ekipazh.json) | spaceship crew | КАСТОР | shipboard AI | [`personalities/ekipazh_ii.txt`](../../personalities/ekipazh_ii.txt) |
| [`personalities/cast_ekipazh.json`](../../personalities/cast_ekipazh.json) | spaceship crew | Юна | intern | [`personalities/ekipazh_stazher.txt`](../../personalities/ekipazh_stazher.txt) |
| [`personalities/cast_garazhi.json`](../../personalities/cast_garazhi.json) | garage hangout | Лёха | mechanic | [`personalities/garazh_mekhanik.txt`](../../personalities/garazh_mekhanik.txt) |
| [`personalities/cast_garazhi.json`](../../personalities/cast_garazhi.json) | garage hangout | Слава | older neighbor | [`personalities/garazh_dyadya.txt`](../../personalities/garazh_dyadya.txt) |
| [`personalities/cast_garazhi.json`](../../personalities/cast_garazhi.json) | garage hangout | Кирилл | visiting student | [`personalities/garazh_student.txt`](../../personalities/garazh_student.txt) |
| [`personalities/cast_teatr.json`](../../personalities/cast_teatr.json) | theater | Степан | actor | [`personalities/theater_actor.txt`](../../personalities/theater_actor.txt) |
| [`personalities/cast_teatr.json`](../../personalities/cast_teatr.json) | theater | Лев Борисович | director | [`personalities/teatr_rezhisser.txt`](../../personalities/teatr_rezhisser.txt) |
| [`personalities/cast_teatr.json`](../../personalities/cast_teatr.json) | theater | Гена | lighting technician | [`personalities/teatr_osvetitel.txt`](../../personalities/teatr_osvetitel.txt) |
| [`personalities/cast_teatr.json`](../../personalities/cast_teatr.json) | theater | Аркадий | critic | [`personalities/teatr_kritik.txt`](../../personalities/teatr_kritik.txt) |
| [`personalities/madhouse_cast.json`](../../personalities/madhouse_cast.json) | mixed example | DC | host, inline prompt | (none, inline) |
| [`personalities/madhouse_cast.json`](../../personalities/madhouse_cast.json) | mixed example | Степан | actor, same persona file as the theater preset | [`personalities/theater_actor.txt`](../../personalities/theater_actor.txt) |
| [`personalities/madhouse_cast.json`](../../personalities/madhouse_cast.json) | mixed example | Лёха | impulsive, inline prompt (a different character from the garage mechanic of the same name) | (none, inline) |
| [`personalities/madhouse_cast.json`](../../personalities/madhouse_cast.json) | mixed example | Сан | curious, inline prompt | (none, inline) |

## Where it lives

| Path | What it holds |
|---|---|
| [`personalities/cast_dopros.json`](../../personalities/cast_dopros.json), `cast_ekipazh.json`, `cast_garazhi.json`, `cast_teatr.json` | the four scenario presets, one `.json` array per scenario, listed in the table above |
| [`personalities/madhouse_cast.json`](../../personalities/madhouse_cast.json) | a shipped example cast mixing file-backed and inline personas; also the filename [`gui/gui_madhouse_cast.py`](../../gui/gui_madhouse_cast.py)'s Save-cast dialog suggests by default, and the fixture [`tests/test_madhouse_cast_voices.py`](../../tests/test_madhouse_cast_voices.py) checks |
| `personalities/dopros_*.txt`, `ekipazh_*.txt`, `garazh_*.txt`, `teatr_*.txt`, `theater_actor.txt` | the persona scripts the cast files point to, one per named role in the table above |

## Constraints

- A `.txt` file is read raw and used as-is for the character's system prompt ([`gui/gui_madhouse_cast.py`](../../gui/gui_madhouse_cast.py)'s `_read_personality`, [`gui/gui_settings_dialog.py`](../../gui/gui_settings_dialog.py)'s `_load_personality`); there is no template or required section structure beyond the three-part brief described above.
- Voice clips are a shared pool, not one-per-persona: the same reference clip backs several characters across different cast files. `tests/test_madhouse_cast_voices.py` only guards `madhouse_cast.json`: every member must name a voice, and no two members of that one file may share one. An empty or unresolved voice falls back to the assistant's own reference clip, which puts two personas in one voice.
- [`personalities/theater_actor.txt`](../../personalities/theater_actor.txt) is shared between `cast_teatr.json` and `madhouse_cast.json`; editing it changes the actor persona in both.

## Coupling

- [`gui/gui_madhouse_cast.py`](../../gui/gui_madhouse_cast.py) lists `personalities/*.txt` for the voice/personality picker combos, loads `cast_*.json` and `madhouse_cast.json` as castable rooms, and is the only writer of [`personalities/madhouse_cast.json`](../../personalities/madhouse_cast.json) (via its Save-cast dialog). [`gui/gui_madhouse_tab.py`](../../gui/gui_madhouse_tab.py), [`gui/gui_madhouse_ui.py`](../../gui/gui_madhouse_ui.py), and [`gui/gui_madhouse_grid.py`](../../gui/gui_madhouse_grid.py) render the resulting character list as the Madhouse room and seat grid.
- [`gui/gui_settings_dialog.py`](../../gui/gui_settings_dialog.py) separately points a single `.txt` from this directory at `ctx.custom_personality_path` / `ctx.custom_personality_text` for the one-assistant personality override; that value feeds the agent graph's personality node, not the Madhouse cast machinery. [`gui/gui_system_info_tab.py`](../../gui/gui_system_info_tab.py) reads `ctx.custom_personality_path` back for display only.
- [`tests/test_madhouse_cast_voices.py`](../../tests/test_madhouse_cast_voices.py) checks [`personalities/madhouse_cast.json`](../../personalities/madhouse_cast.json)'s voice assignments.
