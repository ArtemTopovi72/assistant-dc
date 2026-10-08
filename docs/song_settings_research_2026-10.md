# Why YuE2 ignored the song settings (10-09, 50 searches)

User report: the 🎤 Male button did not reach the song, 🎹 Instrumental came back sung, the
genre drifted. Findings, with what we changed.

## What the model reads

- **Short positive tags, one line.** The official example style is ~10 tags, ~100 chars
  (`City Pop, upbeat, danceable, groovy bass, electric guitar, synth, energetic, joyful, neon city
  night`, models_ext/YuE2-3B/examples). YuE2 Studio's writing rule, taken from the official
  requests: language, genre with its era, vocal (register, gender, delivery), concrete instruments,
  mood in 2-4 words, production, then `N BPM`; no artist names, titles, keys; for an instrumental
  the word `instrumental` where the language goes. YuE (v1) guide: genre, instrument, mood,
  gender, timbre.
- **Negation does not work.** Joint audio-text encoders fail on "with / without vocals"
  (Vasilakis, Bittner, Pauwels, ICASSP 2026, arXiv 2601.13931); Lyria 2 keeps exclusions in a
  separate negative field for this reason. Our caption ended with «no female vocals» and YuE2 got
  the tag with the word *female* in it. **Fixed:** the explicit line names only the chosen voice.
- **Conflicts split the vote.** Suno guides: a vocal or genre tag that another tag contradicts is
  ignored; "the strong tag wins". The writer model added its own genre («Pop») beside the pinned
  one and its own singer («female soprano») beside the button's. **Fixed:** `music.yue2_pin`
  puts the chosen singer first (`male vocal` / `female vocal`), drops tags of the other gender and
  tags that are another genre of the picker; the pinned genre is always present.
- **Text encoders blur gender.** SegTune (arXiv 2606.02638): MuQ-MuLan embeddings of a prompt
  and its gender-flipped copy nearly overlap (cosine distance 0.002). One clear gender tag, no
  contradictions, is the most the prompt can do; the 🎙 Own voice (RVC after the render) decides
  the timbre for sure.

## Instrumental

- No documented instrumental mode in base YuE2. A community LoRA (`ar_lora_inst_v3abc`,
  ~2,700 instrumental tracks, cot=full) is trained on lyrics `[instrumental]` or bare section tags.
- Our bug was ours: the lyric was bare section tags, then the lyric polisher wrote words under
  them. **Fixed:** an instrumental skips the polisher, renders with lyrics `[instrumental]` and a
  style led by `instrumental` with every vocal tag removed; if a vocal is still there (Demucs
  vocal stem over 15 % of the backing's RMS) it is removed.
- To try: the instrumental LoRA (yue2.cpp would need LoRA support), and measure how often the
  base model sings over `[instrumental]`.

## Covers in a star's voice (RVC)

- Epochs ≈ 300 / minutes of clean vocal (community rule); 10-30 min of dry, lead-only vocal;
  pick the checkpoint by ear / by a metric, not the last one; Applio logs mel loss to TensorBoard.
- RVCv2 trained per singer: CER 28 % on M4Singer (Seed-VC EVAL.md) — conversion loses words
  by design; HuBERT carries the words of whatever goes in, so a clear human lead keeps more than a
  YuE take. Night run 10-09 trains on ~15 songs per singer and sings from the original's real lead.
- protect ~0.33-0.5 guards unvoiced consonants; a higher index rate smears diction.

Sources: huggingface.co/m-a-p/YuE2-3B · github.com/timoncool/YuE2-Studio (wiki, docs/mcp-skill.md)
· comfyui-wiki.com/en/news/2026-09-16-yue2-realaudio-encoder · arxiv.org/abs/2601.13931 ·
arxiv.org/pdf/2606.02638 · replicate.com/fofr/yue/readme · jackrighteous.com (Suno ignores
instructions) · github.com/Plachtaa/seed-vc EVAL.md · voicechanger.live/hub/how-to-train-rvc-model
