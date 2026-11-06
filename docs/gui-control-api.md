# Local desktop controls

When the debug chat is enabled, its existing server on `127.0.0.1:8765`
also accepts desktop actions even if the window is hidden. Read the current
token from `runtime/debug_user/gui-control-token` and send it in the
`X-Assistant-Control` header. The token changes each time the app starts.

- `GET /gui/actions` returns the currently supported action names.
- `POST /gui/press` accepts JSON such as `{"action":"voice"}`.

The current allowlist covers `voice`, `pause_speech`, `ultra_search`,
`use_database`, `stop`, and `restart`. Restart closes the app and interrupts
active jobs, just like the window button. Disabled controls return an error; browser-origin
requests and requests without the token are rejected. A timeout means the
action's outcome is unknown, so check the app state before retrying.
