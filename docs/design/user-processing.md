# User request ownership

Telegram intake performs authentication and deterministic routing before durable
acceptance. Commands and navigation do not wait for a semantic model read. Unknown
reply-photo downloads, file requests and armed media intake run in the existing
per-chat worker. Passwords and registration answers never enter the input journal.
Display names accept ordinary nicknames without an LLM validation round trip;
existing length, command and account checks still apply.

Each accepted item keeps its source message, context generation and input-mode
revisions. A prompt opened after an item arrived cannot capture that item or lose
its next answer. Debounce splits adjacent items from different input contexts.
Albums preserve the first arrival context through collection and recovery.
Natural text uses its arrival prefix; it clears the current prefix only while it
still owns that prompt. Navigation remains immediate.

Intent reads belong to a request, rather than to a global text-keyed cache. The
JSON-safe snapshot travels through durable task handoff; locks and live objects
stay in the request runtime. Graph entry starts a new scope unless its caller
explicitly provides the current request. Two users saying the same words cannot
claim each other's classification or attachment context.

Media flows check generation, state and revision after downloads and preparation,
including link and text entry points. Reopening the same mode creates a new
revision. Voice and book uploads use separate temporary source paths. Stop retires
flow revisions and cancels chat-local jobs, so late work cannot reopen a prompt or
publish a cancelled result. Resolver download and classification completions also
check the Stop timestamp before publishing or enqueueing work.
Task preflight rechecks cancellation before invoking the agent; a Stop received
while a language read waits for the model cannot start expensive tools afterwards.
Language reads finish before the session lock is taken; their settings are applied
only if the conversation generation is still current after the read.

Regression coverage includes request-scope isolation, durable recovery, fast prompt
overtaking, album collection, per-voice reply quotes, stale media/link/text
completions, Stop during intake/render and registration without a model. The
existing Telegram, cover and audiobook integrations exercise the whole routing
chain. Live checks use synthetic files and isolated debug users, never real chats.

The debug run verified two simultaneous chats with different code words and the
same recall question, a menu opening while a photo was processed, replacing the
photo, replying to the older photo, sending that exact older photo as a document,
Back navigation and clearing one chat without clearing the other. The sent file
was inspected to confirm it contained the older blue-square fixture rather than
the newer red circles. Private payloads and transcripts stay outside Git.

Legacy archive recovery is described in [chat-context.md](chat-context.md).
Unassigned archives remain preserved until their chat ownership can be verified.
