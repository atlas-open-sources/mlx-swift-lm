# Gemma 4 multimodal test fixtures — sources & licenses

All media used by the Gemma 4 integration tests, with provenance and license.
CC0/Public-Domain files need no attribution; CC-BY files are attributed below.

## Audio

| File | Source | License | Notes |
|---|---|---|---|
| `gemma_audio_librispeech.wav` | LibriSpeech `test`/`dev-clean`, utterance `1272-128104-0000` (via `hf-internal-testing/librispeech_asr_dummy`) | **CC-BY-4.0** | Real human read speech with ground-truth transcript: "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL". 5.86 s, 16 kHz mono. LibriSpeech © Vassil Panayotov et al., derived from LibriVox (public domain). |
| `gemma_speech_test.wav` | Generated locally with macOS `say` + `afconvert` | **CC0 / our own** | "The quick brown fox jumps over the lazy dog near the river bank." Synthetic TTS. |
| `gemma_speech_test2.wav` | macOS `say` | **CC0 / our own** | "She sells sea shells by the sea shore on a bright summer morning." (tongue-twister) |
| `gemma_speech_long.wav` | macOS `say` | **CC0 / our own** | "The weather forecast predicts heavy rain tomorrow afternoon …" (~7 s) |

## Image

| File | Source | License | Notes |
|---|---|---|---|
| `gemma_image_earthrise.jpg` | NASA / Apollo 8, "Earthrise" (Wikimedia Commons `NASA-Apollo8-Dec24-Earthrise.jpg`) | **Public Domain** (NASA) | Earth rising over the lunar horizon. 2400×2400. |

## Video

| File | Source | License | Notes |
|---|---|---|---|
| `gemma_video_bbb.mp4` | Big Buck Bunny (Blender Foundation), 10 s 360p sample via test-videos.co.uk | **CC-BY-3.0** | © Blender Foundation / peach.blender.org. |
| `1080p_30.mov`, `audio_only.mov` | Pre-existing upstream fixtures (added in mlx-swift-lm PR #64) | (upstream) | Color-bar test pattern + tone. Not added by us. |

## Reference transcription baselines (for context, not committed media)

Whisper (large-v3-turbo) transcribes every clip above correctly, including the
synthetic tongue-twister and the LibriSpeech utterance — confirming the audio
files and the Gemma 4 audio pipeline are correct. Gemma 4 E4B's own ASR is much
weaker (audio *understanding*, not transcription); for verbatim transcription use
a dedicated ASR model. Audio-path correctness is therefore asserted via mel
alignment, not transcription accuracy.
