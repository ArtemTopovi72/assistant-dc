# Desktop app (PyQt5)

Start it with `python scripts/launch_all.py`. The window has three full-window pages,
switched by the large tabs at the top. Only one page is shown at a time.

## Interface language

The interface is in English by default. To switch it to Russian, open
**⚙ Settings → Interface → Language** and restart the app. The choice is saved in
`ui_settings.json`. Everything is translated: buttons, tabs, hints, dialogs and status
lines. Any text the table misses is logged to `runtime/untranslated_gui.jsonl`.

Density, UI scale and font scale are in the same Interface section.

## Page 1: Commands

Quick actions are grouped in three columns. Pressing one of them opens the
Conversation page and runs the action there.

| Column | What is in it |
|---|---|
| **Voice & camera** | Talk (push-to-talk), VAD (hands-free listening), camera on/off, capture a frame, whole-frame analysis, voice on/off, pause speech |
| **Images** | Draw, edit, remove an object, remove lettering, upscale, paste from the clipboard |
| **Search & memory** | Web search, deep research, memory profile picker, new profile, compact memory, clear context |

## Page 2: Conversation

The chat takes most of the width, and the **task queue** sits in a column on the right.
You can drag the divider between them.

- **Input line**: press Enter to send. **■ Stop** sits next to **Send** and cancels the
  current answer and any tool it is running.
- **Reply length**: 1 sentence, Short, Auto or Detailed. It only affects the spoken
  reply, not research or image prompts.
- **Queue**: while the assistant is busy, new messages go into the queue instead of being
  lost. Items can be moved up or down, edited, deleted, or sent right away. The queue is
  drained in order when the assistant becomes free.
- Pictures, audio and files can be dropped onto the chat or pasted with Ctrl+V.

## Page 3: Workspace

Tabs can be dragged to reorder them and closed with ✕. Reopen a closed tab from the
layout menu. The layout (open tabs, order, splitter sizes, current page) is restored on
the next start.

| Tab | Purpose |
|---|---|
| **Images** | A strip of every picture made in this session. Click one to open the viewer. |
| **Storyboard** | Plan a picture as labelled boxes, then render it with Ideogram 4. Edits reword one box and reuse the seed, so the rest of the picture stays put. **Plan**, **Draw**, **Redraw**, **New seed**, **Edit boxes**, **Edit in picture**, **Draw with agent** (plan → render → look → fix, up to N rounds). Layouts can be saved and loaded. |
| **Transfer** | Move a garment, a subject, a face or a style from sample pictures onto a Target picture (FireRed). Mark one picture as Target; **Region** lets you draw the exact area. **Protected transfer** keeps the person's identity (mask + paste back). **Quick style transfer** is a one-click preset. |
| **Stress** | Per-word Russian stress overrides for the voice (`+` before the stressed vowel). There is a preview line to hear how a phrase will be stressed. |
| **Music** | Type a topic and get a generated song (YuE2 on ComfyUI). |
| **Weather** | A forecast table for a city and a period, with humidity, wind and a what-to-wear tip from the model. |
| **Mashup** | Put the vocals of one track over the music of another (Demucs stems, tempo and key matching). |
| **Voice Clone** | Clone a voice from a short sample and speak text with it (F5-TTS). |
| **Restyle Video** | Redraw a video clip in a new style while keeping its motion (MiniMax H3 + ControlNet). |
| **Madhouse** | Several AI characters talking to each other, and to you. |
| **Search** | Results of the last web search. |
| **Research** | Deep Research: pick a depth profile, run it, and read the cited report. The sources and quotes are verified. |
| **Database** | Your document library for RAG: add files, build the hybrid index (BGE-M3 + reranker), and set *use database* and *top-k*. |
| **Memory** | Memory Center: the facts the assistant keeps about you, profiles, summaries and compaction. |
| **Model** | The chat model in LM Studio, its context length and sampling, and a reload. |
| **Status** | Diagnostics: the model, voice, microphone, web search, and optional packages (✓ / ✗). |
| **Log** | The app's live log. |
| **Telegram** | Bot control panel (see below). |
| **Admin** | Operator tools: users, quotas, maintenance. |
| **Code** | The code sandbox: pick a working folder (yours or a chat's), browse and drop files, ask for work in plain words, reset the folder. |
| **Characters** | Build a character LoRA from photos (prepare → train → generate) and choose whether it is offered in Telegram. |

## Telegram tab

- **Bot token**: from @BotFather. It is stored in Windows settings, not in the project.
- **api_id / api_hash** (optional): when both are set, the app starts a local
  `telegram-bot-api --local`, which allows files up to 2 GB.
- **Admin chat IDs**: these users are approved as admins on first contact.
- **Default bot language** (Russian / English): the language for users who have not
  picked one. It is written to `.env` as `TG_DEFAULT_LANG`.
- **Reply with voice**, **Start with the app**, **Quiet mode** (no online/offline notices).
- **Active chats** shows each user's current stage, and there is a **queue** readout.
- **Users** table: approve, reject, ban, make admin, remove admin, sandbox level
  (off / files / code / host), reset a user's folder, restore a user-database backup.
- **Live log**, with clear and export.

## Settings dialog (⚙)

Model and reasoning depth, reply length, interface (scale, density, font, language),
disable web search, fact extraction, real-person photo reference, disable microphone,
disable TTS, voice and personality files, web search region and max results.
