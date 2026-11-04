# Subarashii

Dockerized Japanese / Chinese (Mandarin) / Korean / Russian / Ukrainian → English dubbing. Hand it a video; get back the same video with an
English dub track in which every character keeps **their own cloned voice**, delivered with the
**emotion of the original line**, and timed to fit the original mouth movement.

The spoken language is detected automatically (from the audio track's language tag, otherwise
from the speech), or set it per job: the web form's *Spoken language*, `--lang ja|zh|ko|ru|uk` on the
CLI, or `input.language` in the config. Per-language settings (second speech recognizer, timing
ratio, known recognizer hallucinations) are under `languages:` in the config.

## How it works

| # | Stage | What happens | Model |
|---|-------|--------------|-------|
| 1 | ingest | probe the video, extract the original-language audio track | ffmpeg |
| 2 | separate | split dialogue from music/SFX | BS-RoFormer (audio-separator) |
| 3 | transcribe | VAD cuts utterances at pauses, a voice-change check splits back-to-back speakers, crosstalk (two people at once) is split into one line per voice, then each utterance is transcribed | Silero VAD, ECAPA, pyannote segmentation-3.0 + MossFormer2 (crosstalk), faster-whisper large-v3 + a per-language fine-tune: litagin/anime-whisper (ja), BELLE-2/Belle-whisper-large-v3-zh-punct (zh), ghost613/whisper-large-v3-turbo-korean (ko), antony66/whisper-large-v3-russian (ru), large-v3 alone (uk); Whisper language ID when the language is auto |
| 4 | diarize | group lines into characters by voice, then re-join same-speaker sentence fragments and re-transcribe them whole | SpeechBrain ECAPA + agglomerative clustering |
| 5 | profile | per character: gender/age/pitch/pace estimates, best clean clips → voice reference, scene pictures; per line: original clip | audeering wav2vec2 age-gender, librosa |
| 6 | proofread | LLM picks between the two recognizers' versions of each line using context | local LLM (llama.cpp) or Claude |
| 7 | characters | LLM reads the labelled transcript: names (only when the dialogue makes them clear, otherwise a description like "Old monk"), gender, age, personality, speech style, glossary | same LLM |
| 8 | translate | LLM writes **three** dub renderings per line (natural / concise / minimal) against a syllable budget derived from the line's duration and how fast it was spoken | same LLM |
| 9 | check | LLM editor reads the English script scene by scene and flags lines a viewer would misunderstand (wrong meaning, misheard dialogue, a reply that doesn't follow, a name rendered two ways). Those are re-translated with the problem explained, and a blind A/B judge must prefer the new line before it's kept. Changed scenes are read again, up to `check.rounds` passes; lines it still objects to after a rewrite are marked ⚠ in the web UI. A rewrite may not bring in a name that isn't in the original line | with llama.cpp: `check.llamacpp_model` (qwen3.5-35b-a3b), a stronger critic than the translator |
| ⏸ | review | optional pause to fix characters / lines in the web UI | |
| 10 | synthesize | IndexTTS-2.5 clones the character's voice, using the original line as the emotion reference; fitting loop below | IndexTTS-2.5 |
| 11 | mix | music/SFX + English lines + original non-speech vocals (gasps, laughs, screams) | |
| 12 | export | mux: English track (default) + original-language track + English subtitles | ffmpeg |

**Voice matching.** Instead of picking a stock voice that might suit the character, each character's English
voice is cloned from the original actor. A burly man stays deep and gravelly, and a child stays a
child. The emotion comes from the line itself: IndexTTS-2.5 takes the original clip as an emotion
reference, so a shouted original line produces a shouted English one.

**Timing fit.** Each line gets a time slot: its original span, which can extend up to `timing.max_overflow`
seconds into the silence before the next line. Synthesis tries, in order:

1. the natural line at normal speed;
2. the same take, sped up with IndexTTS's `duration_factor` (no more than 1.25× by default);
3. the concise version, then the minimal one;
4. an LLM rewrite to a measured syllable budget, using that character's actual TTS pace;
5. a rubberband time-stretch of up to 1.1× as a last resort, after which the line is flagged as `overflow`
   in the UI.

When a take is much shorter than the original, it's slowed down slightly to fill the slot more naturally.

## Requirements

- NVIDIA GPU with 12 GB or more of VRAM and the NVIDIA
  Container Toolkit.
- About 19 GB for the Docker images (they share a torch base layer) and about 25 GB under `ASSETS_DIR` for
  model weights.
- A translator:
  - **Local (default):** a llama.cpp server in router mode (`--models-preset`). Subarashii calls
    `/models/unload` to free VRAM before the ASR and TTS stages. Any OpenAI-compatible server works with
    `LLM_PROVIDER=openai`, but it won't be unloaded automatically.
  - **Claude:** set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`. It uses server-side refusal fallback
    (`fallbacks: "default"`).
  - The provider, endpoint, API key and model can also be set on the web UI's **Settings** dialog (gear icon, top right). They are saved in
    `ASSETS_DIR/data/settings.json` (readable only by you) and override `.env` and `config.yaml`.

## Setup

```bash
cp .env.example .env          # set ASSETS_DIR, LLAMACPP_API_KEY (or ANTHROPIC_API_KEY)
docker compose build          # ~15 min first time
docker compose up -d          # web UI on http://localhost:7860
```

Model weights download on first use into `ASSETS_DIR/models`.

## Usage

**Web UI** at http://localhost:7860: upload a video (or pick one from `ASSETS_DIR/data/input`) and click
Start. It runs fully automatically, and two progress bars show the overall progress with an
estimated time left, plus the current step. You don't need to know the original language or the source material.

You can optionally tick *"Let me check characters and lines before voicing"*. The job then pauses after
translation and shows:
- a card for each character, with a voice sample and pictures from scenes where that voice is heard. Use
  *Same person as…* to combine duplicates, and type in names you know;
- the English script, with ▶ Original / ▶ English buttons, other shorter or longer versions of each line
  you can pick with one click, and ⚠ flags on lines the system was unsure about ("Only lines that need
  attention").

Click **Approve & voice it**. After the job finishes you can still edit lines and click
**Re-render changed lines**: only those lines are re-voiced.

**CLI** (fully automatic):

```bash
docker compose run --rm app subarashii dub video01.mkv           # file in ASSETS_DIR/data/input
docker compose run --rm app subarashii dub video.mp4 --provider anthropic
docker compose run --rm app subarashii dub video.mp4 --set translation.honorifics=keep --set timing.max_overflow=0.7
docker compose run --rm app subarashii dub video.mp4 --review          # pause; finish in the UI or with resume
docker compose run --rm app subarashii resume <job-id> --from-stage translate
docker compose run --rm app subarashii jobs
```

The output is written to `ASSETS_DIR/data/output/<name>.en-dub.<mp4|mkv>`. Each job's working files
(stems, per-line clips, `project.json`) are in `ASSETS_DIR/data/jobs/<id>/`.

## Configuration

Defaults are in [subarashii/default_config.yaml](subarashii/default_config.yaml). Override them in
[config/config.yaml](config/config.yaml), per job with `--set key=value`, or in the UI's "Advanced
settings". The knobs you'll use most:

| Key | Default | Effect |
|-----|---------|--------|
| `llm.llamacpp.model` | `qwen3-30b-a3b` | translator model (any id from the router's `/models`) |
| `translation.honorifics` | `drop` | `keep` for Japanese -san / -kun / senpai, Korean -ssi / -nim, Russian / Ukrainian name-and-patronymic and diminutives |
| `check.rounds` | `2` | script-check passes (`check.enabled=false` skips the check) |
| `check.llamacpp_model` | `qwen3.5-35b-a3b` | model the script check uses with llama.cpp (`""` = the translator's) |
| `diarization.distance_threshold` | `0.62` | lower → more distinct characters |
| `diarization.max_speakers` | `0` | force a character count if you know it |
| `overlap.enabled` | `true` | split crosstalk (two people at once) into one line per voice |
| `overlap.scan` | `true` | also separate all speech to find quiet crosstalk pyannote misses (adds a few minutes per episode) |
| `asr.speaker_change_threshold` | `0.85` | lower → split utterances at voice changes more eagerly |
| `timing.max_overflow` | `0.5` | seconds a line may run past its original end |
| `timing.min_duration_factor` | `0.8` | fastest allowed TTS speed-up |
| `mix.min_speech_over_bg_db` | `6.0` | keep dubbed dialogue at least this far above music/effects |
| `tts.emo_alpha` | `0.75` | how strongly the original line's emotion transfers |
| `mix.keep_nonspeech_vocals` | `true` | keep the original gasps, laughs and screams |

## GPU memory

Only one large model is resident at a time. Before each GPU stage, Subarashii unloads the TTS model and
every model loaded in the llama.cpp router. If something else uses the same router while a job runs, a
model can be reloaded mid-job and cause CUDA OOM. In that case, re-run the job from the failed stage.

## Known limits

- **Overlapping speech:** when two characters talk over each other, the crosstalk is split into two voices
  and each gets its own line and dub (marked ⚠ in the review UI, since separated audio is harder to
  transcribe). A split is only kept when it looks like two people (different voices, different words), and
  it is undone if both halves turn out to be the same character. Three or more people at once, or
  crosstalk longer than `overlap.max_window` (10 s), still goes to the dominant voice. Now and then the
  separator swaps the two voices partway through, which garbles both lines; the job log's `crosstalk:`
  line says how many places were split and why others were not.
- **Speaker clustering:** a character who whispers in one scene and shouts in another can be split in two,
  and siblings or characters voiced by the same actor can be merged. Use Merge or "+ new char" in the UI.
- **Short lines:** interjections under 1 s are assigned to the nearest voice and are the most likely to be
  wrong.
- **Songs:** Song lyrics (e.g. opening/ending themes) are flagged by the LLM and left in the original language. Check the lyric boxes if it misses
  some.
- **Chinese and Korean** are newer than Japanese and less tested. Chinese support targets Mandarin; Cantonese is detected as Chinese but
  transcribed less accurately. Spoken Chinese gives no dialogue cue for a
  character's gender, so it comes from the voice analysis and the LLM only.
- **Russian and Ukrainian** are the newest and haven't been tested on real episodes yet; their speech
  recognizers and timing ratio were chosen on read speech (FLEURS). Ukrainian uses large-v3 alone: the
  only Whisper fine-tune recognized less accurately. Whisper can confuse the two on little
  speech, so the log shows the runner-up when it's close; set the language explicitly for short clips or
  dialogue that mixes them.
- **No lip re-animation:** the video is never altered; sync comes from timing and phrasing only.

## Credits and licenses

Subarashii is made by CheyOps Research and is MIT licensed (see [LICENSE](LICENSE)). If you use, fork or
build on it, a credit such as "Based on Subarashii by CheyOps Research" is appreciated, though not required.

It builds on these projects; apart from pyannote
segmentation-3.0, model weights are not included and download on first use under their own licenses, so check each one before commercial use
(some, such as the audeering age/gender model, are non-commercial).

- [IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5): voice cloning and synthesis
- [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) (and [large-v3 turbo](https://huggingface.co/openai/whisper-large-v3-turbo)'s generation config for Korean) and [faster-whisper](https://github.com/SYSTRAN/faster-whisper): speech recognition
- [anime-whisper](https://huggingface.co/litagin/anime-whisper), [Belle-whisper-large-v3-zh-punct](https://huggingface.co/BELLE-2/Belle-whisper-large-v3-zh-punct), [whisper-large-v3-turbo-korean](https://huggingface.co/ghost613/whisper-large-v3-turbo-korean), [whisper-large-v3-russian](https://huggingface.co/antony66/whisper-large-v3-russian): per-language fine-tunes
- [SpeechBrain ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb): speaker embeddings
- [pyannote segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) (MIT, © 2023 CNRS): crosstalk detection.
  Its weights are included in [models/pyannote-segmentation-3.0](models/pyannote-segmentation-3.0) with their license
- [MossFormer2_SS_16K](https://huggingface.co/alibabasglab/MossFormer2_SS_16K) from [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
  (Apache-2.0): separating two voices that talk at once. Its network code is included in
  [subarashii/vendor/mossformer2_ss](subarashii/vendor/mossformer2_ss) with its license
- [audeering wav2vec2 age/gender](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender): voice profiling
- [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) (BS-RoFormer): dialogue separation
- [Silero VAD](https://github.com/snakers4/silero-vad): voice activity detection
