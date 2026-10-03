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
- **🎨 Draw**: generate an image, edit an image, remove an object, remove lettering,
  regenerate, analyze a photo, 📐 size and aspect ratio.
- **📊 Presentation**: build or edit a .pptx deck. The figures in it are fact-checked
  against the sources.
- **🎵 Songs / 🎛 Song settings**, **🎚 Mashup**, **🎤 Cover**.
- **📁 Sandbox**: your working folder.
- **🎬 Long video recap**, **🎨 Restyle video**, **🎞 Animate photo**, **🎙 Clone voice**.
- **🧑 Characters**: draw with a trained character LoRA.

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
| `/lang` | Change language |
| `/docs` | Your indexed documents |
| `/facts` | What the assistant remembers about you |
| `/files`, `/sandbox`, `/reset_sandbox` | Working folder (for users who have sandbox access) |

## Queue and limits

All users share one GPU through a fair queue (Redis if it is available, otherwise in
memory). Each request shows its place in the queue and an ETA, plus a **⛔ Cancel this
request** button. Quotas and queue limits are set in `.env`; see `.env.example`.
