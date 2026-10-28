# Chat context implementation plan

**Goal:** Answer the current request using its own attachments while retaining useful conversational history and user facts.

**Architecture:** Keep the existing durable `_Task`, image register and per-chat session store. Capture transport identity before debounce; resolve a canonical attachment once; commit memory deltas under the session lock. A cleared session rejects commits from older generations.

**Tech stack:** Existing Python, Telegram transport, LangGraph and Gemma; no additional database or memory framework.

## Implementation

- [x] Input identity: stamp message IDs on durable incoming items and albums; replies use the request's own message, excluding navigation/quoted fragments.
- [x] Vision routing: questions cannot become photo-intake prompts; explicit image references outrank filesystem timestamps; missing explicit references never silently select another image.
- [x] Attachment context: clear inherited GUI references, capture selected image IDs before queue execution, use register provenance rather than filename prefixes.
- [x] Memory commits: preserve concurrent additions; older tasks cannot replace newer current-media state; clearing removes active media and invalidates old commits.
- [x] Memory composition: retain recent dialogue and saved facts; identify image observations by attachment and time, separating current evidence from historical observations.
- [x] Verify offline regressions, then live debug sequences (caption, question before/after photo, forwarded quotes, replacement photo, album, explicit old-image reply, memory recall/update/clear). Review and restart before committing.

## Evidence behind the choices

- [Context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): curate relevant context, keep references, compact redundant material rather than indiscriminately expanding prompts.
- [Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory): conversation-scoped state and explicit history management.
- [Long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory): durable facts separated by namespace from individual conversational turns.
- [LongMemEval](https://arxiv.org/abs/2410.10813): test temporal changes, recall, knowledge updates and abstention, rather than only single-turn responses.
- [Telegram Bot API](https://core.telegram.org/bots/api): captions, reply message IDs, media groups and update IDs are transport data, not facts inferred by a language model.

The 52 distinct search queries and applied principles are recorded in [chat-context-research.json](chat-context-research.json). Full discovery results are kept locally in `.claude/.scratchpad/chat-context/research.json`. Research does not justify adding a knowledge graph or replacing the existing model; the observed failures are ownership, routing and state-commit defects.

## Ownership and compatibility

Telegram archives now live under each chat's memory directory and clear generation. A late write from an older generation cannot be recalled by the new conversation. GUI archives retain their existing namespace. Legacy global archives have no reliable owner metadata and are not exposed to Telegram as a fallback; existing files are preserved on disk.

Verified legacy records can be restored with `scripts/migrate_context_archives.py` and an explicitly reviewed private ownership manifest, while the bot is stopped. The tool previews by default, verifies IDs/content hashes and generation, refuses conflicting destinations, backs up sessions and preserves originals. Active records get a compact historical catalog in the existing working memory; records preceding a chat clear are kept only in that owner's inactive `legacy_archive` directory.

The local migration verified nine records against unique chat ownership and at least two distinct full outgoing replies (or current generated memory references). Eight were restored to active namespaces, one predating a clear was preserved offline, and 24 files without sufficient ownership evidence remain untouched. All 33 original files and a session backup were retained. Private evidence/manifests/reports stay outside Git. Six migration regressions verify preview, ownership isolation, original/history preservation, idempotence, retired records, hash/conflict checks and stale-generation refusal.

Image prompts belong to their register entries. A new user photo does not inherit a previous generated image's prompt. Current image identity is persisted with the session. Facts and recent dialogue remain available until explicitly cleared; clearing also invalidates queued work, ASR results, retellings, research and weather history commits from the previous generation.

## Live verification

The running Gemma model was exercised through the debug user's normal dispatch, debounce, queue and delivery paths with synthetic fixtures:

| Sequence | Observed result |
| --- | --- |
| Photo with question | Answered the question instead of asking what to do with it |
| Blue squares, then red circles, then follow-up | Answer referred to the replacement image |
| Reply to the older photo | Answer used the older photo |
| Question, forwarded photo, forwarded comments | Answered the user's question; quoted comments were not treated as instructions |
| Two-image album | Distinguished both images and their counts/colors |
| Captionless photo, then question | Follow-up used that photo |
| Photo followed by Back | Reply anchor remained the photo's message ID |
| Clear during photo processing | Next request did not recover the old image |
| Remember, unrelated question, recall, update, recall | Recalled the fact and then the newer value |
| Clear, then request the former fact | Admitted the fact was unknown |
| Recall archive in its own chat | Retrieved the synthetic control code |
| Recall the same archive in another chat or after clear | Did not reveal the control code |
| Question followed by a forwarded video, then a follow-up | Answered from video frames, replied to the user's question, retained the video for the follow-up and omitted photo-editing buttons |

Private debug transcripts and outbox payloads are kept outside Git under `.claude/.scratchpad/chat-context/`.

Final offline verification: 87 pytest tests across the context contract, incoming durability, context v2, fact selection/injection, persistence faults and the no-global-module-patching gate. Standalone Telegram pipeline, forwarded-media, image delivery/targeting, style, weather, account and picture suites also passed. The targeting suite needs a 240-second runner timeout because its real debounce scenarios take about 143 seconds.
