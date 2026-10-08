# Cover lyrics: words laid on an existing tune with meaningful repeats (2026-10-08)

The problem: a cover re-sings a song with new words (`media/remix.py` → `resing`). The
old `_layout` only picked and dropped lines to fit each section's note count. Repeats came
out wherever the model happened to put them. When a section had notes to spare they stayed
empty, and lines that did not fit were dropped. We want the repeats placed the way the user heard
them in the «3 сентября» / «Я русский» covers: key words repeated on purpose, the hook at the
edges of the chorus, the refrain intact.

50 web searches were run. This file keeps what was learned and what was built from it.

## What the research says

**Song translation / adaptation theory**
- Low's *Pentathlon Principle*: singability, sense, naturalness, rhythm, rhyme, traded against
  each other and never all maxed. Low separates translation, adaptation (keeps theme)
  and replacement (new text on the old tune; a parody or cover with new words is this).
- Franzon: the choices are to leave the song untranslated, translate the words without regard
  to the music, write new words for the melody, or change the music to fit the words. Ours is
  "new words, melody may bend": YuE2 re-sings on the full score.
- Apter & Herman, *Translating for Singing*: rhythm, syllable count, vocal burden, rhyme,
  **repetition** and sound are each separate problems.
- Mkhitaryan (2021, 272 Russian covers): equirhythmic techniques are generalization,
  omission, expansion, integral transformation and change of grammar. The sound
  (equiphony) matters a lot.
- Translators tend to *avoid* repetition and so lose its function (Babel, Klingberg).
  The ATA guide says to keep repeats because they carry the message.
- **Kim & Goto, ISMIR 2023**: across languages the self-(dis)similarity matrices of sections
  look alike when a song is translated singably. So *matching the original's map of
  repeats* is a measurable sign of a good adaptation. This is the core of the change.
- A singable-translation evaluation (arXiv 2308.13715): the degree of phoneme repetition per
  section stays the same across languages.

**Songwriting craft**
- Repetition is what makes a hook stick (Schubert 2023 review: an earworm needs repetition in
  the music). Choruses repeat; verses tell the story with new details each time.
- Title/hook placement: at the start of the chorus, at its end, or both ("bookend"). A
  4-line chorus often has the title in lines 1 and 4. Repeat the hook 2–4 times. Varied
  repetition (one word changed in the last chorus) keeps it from going flat.
- Repeat with meaning: repeat KEY words, never function words. Figures: anaphora (same
  start), epistrophe (same end), epizeuxis («word, word, word»), anadiplosis (end of one
  line opens the next).
- Pattison: stressed syllables on strong beats. Meaning can promote a word to a strong beat;
  even/odd line counts read as stable/unstable.
- Important words go on long or high notes, and held notes want open vowels. Avoid
  consonant clusters and stops at word ends on held notes (sung intelligibility studies: ~3/4 of
  errors are consonants, codas worst).
- Non-lexical vocables («ой-ой-ой», «la la») are legitimate musical material. Keep them
  where the original or the user's lyric has them.
- Weird Al: analyse the original's syllables, rhymes and form first, brainstorm many variants
  per line, keep as close to the original's shape as possible, and never cram syllables.

**LLM methods**
- LLMs are bad at exact syllable counts. Tokenization hides syllables (one study: about 55%
  of lines hit the count at best; ChatGPT-4 had about an 80% syllable error rate against a
  fine-tuned model's 4%). Cyrillic splits into byte pieces. **Count with code, not the model.**
- ICCC 2025: a write → check → feedback loop fixes meter, and **rule-based feedback beats an
  LLM judge**.
- Over-generate and rank (best-of-N with a scorer) reliably lowers the share of bad outputs.
- REFFLY (NAACL 2025): revise a draft into a melody instead of writing from scratch. Put
  content words (nouns, verbs, adjectives) on prominent notes.
- Ou et al., ACL 2023: prompt-based length, rhyme and word-boundary control. Word
  boundaries should fall on melody rests.
- Liu et al., EACL 2026 SRW: moderately complex multi-step prompts make lyrics sound most
  natural when sung.
- Riedl, *Weird AI Yankovic*: generate backward from the rhyme word, keep syllables and
  rhyme scheme, and filter out lines with the wrong count.
- KeYric: a keyword skeleton first (plan), then lines, for better coherence.
- Russian tools: RUAccent (already ours), Koziev's Russian Poetry Scansion Tool and RIFMA
  dataset for checking stress and rhyme.

## What was built (media/remix.py)

1. `_repetition_profile(phrases)`: reads how the ORIGINAL vocal repeats itself (Whisper
   phrases): refrains (lines that come back, and how many times), a key word sung several times
   in a row, anaphora, sung interjections. The model gets the *shape*, not the words.
2. `_hook_words(lines)`: the user's hook is the one or two most sung content words of
   their lyric (the song's subject).
3. `_arrange(...)`: after the line placement, the model re-arranges each sung section using
   **only the user's own words** (the section's lines, the hook, unused lines for verses).
   - It is told per section how many notes are spare or how many syllables are too many
     (`_need`).
   - It fills spare notes by repeating the hook, puts the hook at the start and end of the
     chorus, keeps choruses identical, and cuts the least important words.
4. `_arrange_score(...)`: deterministic, rule-based check of each draft.
   - Hard rules: no foreign words and at most +25% syllables.
   - Costs: distance to the note budget (crowding counts double), lost content words
     (repeats must not eat the meaning), choruses that differ, a chorus without the hook.
   - `ARRANGE_TRIES` (3) drafts at rising temperature. The best is kept only if it beats the
     plain placement.
5. Glued lines («… ой-ёй-ёй. Вроде стали …») are split back into one sung phrase per line.

Live check on Gemma 4 26B with a user lyric on the «3 сентября» score:
- Plain placement: verse 2 got 19 of 33 notes, and the dropped lines were never sung.
- Arranged: verse 2 takes the next unused line, the pre-chorus ends on the hook ×3, and the
  final chorus closes on the hook. Cost went from 1.45 to 1.12.

## Sources
- Low, Pentathlon: https://ojs.letras.up.pt/index.php/tm/article/view/14857
- Franzon, five choices (via Hsu): https://www.ciol.org.uk/make-it-sing
- Apter & Herman, *Translating for Singing*: https://www.abebooks.com/9781472571885/Translating-Singing-Theory-Art-Craft-1472571886/plp
- Mkhitaryan 2021: https://kpfu.ru/uz-eng-hum-2021-1-7.html
- Kim & Goto ISMIR 2023: https://staff.aist.go.jp/m.goto/PAPER/ISMIR2023POSTERkim.pdf
- Singable translation evaluation: https://arxiv.org/pdf/2308.13715
- Repetition in translation (Babel): https://lwc1.benjamins.com/catalog/babel.00093.kli
- Earworms review: https://www.unsw.edu.au/newsroom/news/2023/04/ear-resistible--why-there-are-some-songs-we-simply-can-t-get-out
- Hook placement: https://rhymebook.com/guides/how-to-write-a-chorus , https://tonyconniff.com/?p=3357
- Changing a chorus: https://blog.sonicbids.com/3-reasons-you-shouldnt-change-up-your-chorus-lyrics
- Repetition figures: https://literarydevices.net/anadiplosis
- Lyric setting / Pattison: https://en.wikipedia.org/wiki/Lyric_setting , https://www.jmcacademy.edu.au/news/top-lyric-writing-tips-with-pat-pattison
- Vowels on held notes: https://fastrhymes.com/blog/how-to-write-songs-that-are-fun-to-sing-crafting-vocalist-friendly-melodies
- Sung intelligibility: https://d-scholarship.pitt.edu/19444/1/EMR000050a%2DCollister%2DHuron.pdf
- Vocables: https://en.wikipedia.org/wiki/Vocable
- Weird Al process: https://graduateschool.aub.ac.uk/top-news/how-weird-al-yankovic-removed-the-misogyny-of-blurred-lines-by-adding-grammar-lessons.html
- Syllable-count weakness: https://ontologic.substack.com/p/llm-powered-lyric-generation , https://arxiv.org/pdf/2409.00292
- ICCC 2025 feedback loop: https://computationalcreativity.net/iccc25/papers/iccc25-agirrezabal2025refining.pdf
- Best-of-N: https://arxiv.org/pdf/2406.16838
- REFFLY: https://arxiv.org/abs/2409.00292v2
- Ou et al. ACL 2023: https://arxiv.org/pdf/2305.16816
- Liu et al. EACL 2026 SRW: https://aclanthology.org/2026.eacl-srw.42
- Weird AI Yankovic: https://arxiv.org/pdf/2009.12240
- KeYric: https://smcnus.comp.nus.edu.sg/archive/pdf/2024/2024_KeYric.pdf
- Russian scansion: https://arxiv.org/html/2502.20931v1
- SongComposer: https://aclanthology.org/2025.acl-long.352
- Unsupervised melody-to-lyric: https://arxiv.org/pdf/2305.19228
