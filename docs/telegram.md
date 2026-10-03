# Telegram bot

Set it up in the desktop app's **Telegram** tab (see [gui.md](gui.md#telegram-tab)).
New users register with a name and a password, then wait for an admin to approve them.
Admins can approve users in the Telegram tab, or right in Telegram if their chat ID is
on the admin list.

## Language

The interface is Russian by default (`TG_DEFAULT_LANG` in `.env`, also set from the
Telegram tab). A user can switch with `/lang`, **⚙️ Settings → 🌐 Language** or
**👤 Account**, and their choice is never overwritten. The `/` command menu is
registered in both languages.

## Main keyboard

| | | |
|---|---|---|
| 🎨 Creativity | 🔎 Search | 👤 Account |
| 📝 Feedback | 🌤 Weather | ❓ Help |
| ⚙️ Settings | 🛒 Ozon | ⛔ Stop |

Reply keyboards are persistent: they stay open after a button is pressed.
Any plain message goes to the agent. It chooses the tools itself: drawing, search,
documents, code and so on.

### 🎨 Creativity
| | |
|---|---|
| 🖼 Images | 🎶 Music |
| 🎬 Video | 📊 Presentation |
| 📁 Sandbox | |

- **🖼 Images**: 🎨 Draw (generate, edit, remove an object or lettering, regenerate,
  analyze a photo, 📐 size), 🎭 Change style, 🧑 Characters (a trained character LoRA).
  Photo or drawing is decided by your words in the current message: a photo unless you
  name a drawn style; «ещё раз» repeats the look of the request before it.
- **🎶 Music**: 🎵 Songs, 🎚 Mashup, 🎛 Song settings, 🎤 Cover (a file or a YouTube link,
  then optional new words), 🎙 Clone voice, ✨ Improve lyrics, ✍️ Write lyrics.
  In 🎵 Songs a wish becomes a written and polished lyric before singing; ready words get
  a choice: sing them as they are or ✨ polish first (the changes are shown).
  ✨ / ✍️ answer in two messages: what was changed, then the lyric alone to copy, with
  🎵 Sing it and ✨ Again.
- **🎬 Video**: 🎬 Long video recap, 🎨 Restyle video, 🎞 Animate photo. An animated
  photo is the clip's first frame and keeps its look; the camera moves only when asked.
- **📊 Presentation**: build or edit a .pptx deck; the figures are fact-checked against
  the sources.
- **📁 Sandbox**: your working folder.

While the bot waits for your next message (a topic, a photo, a name), its prompt carries
**✖️ Cancel**. **↩ Back** under a message only takes its buttons away; the message stays.

### 🔎 Search
🔍 Web search, 🔬 Deep research (a cited report), 🎚 Depth (shows how long each level takes).

### 🌤 Weather
Now, 24 h, 48 h, a specific date or another city, plus a what-to-wear tip.

### 🛒 Ozon
Find a product, cheapest, best rated, open a product card from a link, review summary,
compare, fast delivery, shopping list, city or pickup point, build a set.

### ⚙️ Settings
| | |
|---|---|
| 🗣 Reply format | 🔔 Notifications |
| 📌 Remember | 🗑 Clear chat |
| 📚 Documents | 🧠 My facts |
| 🌐 Language | 📊 Status |
| 🧠 Reasoning | 📎 Photo as file |
| 🗣 Assistant voice | |
| 🔐 Admin panel *(admins only)* | |
| ↩ Back | |

**🗣 Assistant voice** opens the voice library (**📚 My voices** inside it lists them all): every sample you
send (for the assistant, a clone or a clip) is kept; ▶️ listen, ✏️ name it to keep it
for good, 🗑 delete it. The last unnamed ones are kept as recent voices.

**🗣 Reply format** opens a small inline picker with three choices: text, text + voice,
or voice. The current choice is marked ✅, and the picker updates in place without
closing the menu.

## Commands

| Command | What it does |
|---|---|
| `/start` | Welcome and keyboard |
| `/help` | List of commands |
| `/draw`, `/img` | Generate an image |
| `/search` | Web search |
| `/deck` | Build or edit a PowerPoint deck |
| `/clear` | Clear conversation history |
| `/voice` | Toggle voice notes |
| `/cancel` | Cancel the current flow |
| `/size` | Image size and aspect ratio |
| `/depth` | Research depth and its duration |
| `/settings` | Settings |
| `/account`, `/setname`, `/setpassword` | Account |
| `/subscribe`, `/unsubscribe` | Bot online/offline notices |
| `/feedback` | Bug report or feature request |
| `/status` | Service health, queue and your usage |
| `/lang` | Change language (also ends «answer in English from now on»; so does `/clear`, or 6 h of quiet) |
| `/docs` | Your indexed documents |
| `/facts` | What the assistant remembers about you |
| `/files`, `/sandbox`, `/reset_sandbox` | Working folder (for users who have sandbox access) |

## Queue and limits

All users share one GPU through a fair queue (Redis if it is available, otherwise in
memory). Each request shows its place in the queue and an ETA, plus a **⛔ Cancel this
request** button. Quotas and queue limits are set in `.env`; see `.env.example`.
