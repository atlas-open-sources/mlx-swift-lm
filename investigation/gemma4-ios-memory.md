# Gemma 4 E2B — iOS multimodal memory investigation

> **Scope:** Why Gemma 4 multimodal (image/audio/video) crashed on iPhone, what the
> real limits are, and how to gate it. This folder lives only on our
> `gemma4-integration` fork branch — **exclude it from any upstream PR**, but the PR
> description may link here for the rationale behind the iOS-specific changes.

**Model:** `mlx-community/gemma-4-E2B-it-qat-4bit` (Gemma 4 "E2B", QAT 4-bit; loads vision + audio towers).
**Devices:** iPhone 16 Pro Max (8 GB, A18 Pro) · iPhone 17 Pro Max (12 GB, A19 Pro). iOS 26.3.x.
**Build:** `Seek Deep Local AI (iOS App Store)` scheme, Debug, `increased-memory-limit` entitlement present & honored.
**Date:** June 2026.

## TL;DR

1. **The per-app memory ceiling is a flat ~6144 MB on BOTH phones** — Apple caps a
   single iPhone app independent of physical RAM. The 17's extra 4 GB is unusable by
   one app. This is why the 8 GB and 12 GB phones behaved *identically*.
2. **The crashes were caused by the MLX GPU buffer cache, not the RAM or the model.**
   A 512 MB `Memory.cacheLimit` hoarded the exact headroom multimodal prefill needed.
   **Capping it at 64 MB on iOS reclaimed ~440 MB and fixed everything** — combined
   media started working and video went from 4 → 16 frames, with *no* throughput loss.
3. **Gate on a token budget, not frame count.** Robust prefill budget ≈ **1000 tokens**.

## The ceiling (measured via `os_proc_available_memory()`)

`availBeforeJetsam` = bytes the process can still allocate before the kernel jetsam-kills it.
`ceiling ≈ phys_footprint + availBeforeJetsam`.

| Device | RAM | Ceiling | Loaded model footprint | Inference headroom |
|---|---|---|---|---|
| iPhone 16 Pro Max | 8 GB | **6144 MB** | ~4375 MB | ~1768 MB |
| iPhone 17 Pro Max | 12 GB | **6144 MB** | ~4400 MB | ~1740 MB |

Entitlements: `increased-memory-limit` honored (8 GB phone gets 6 GB = 75%, above the
~50% un-entitled default → proof it works). `increased-debugging-memory-limit` +
`extended-virtual-addressing` are **stripped at signing** (App ID/profile lacks the
capability). `DirectDownload.entitlements` lacks `increased-memory-limit` entirely.

## Root cause: the GPU buffer cache

`LLMEvaluator.configureCacheForModel` set `Memory.cacheLimit = 512 MB` for small models.
That cache is part of the ~4.4 GB load-time footprint and competes directly with prefill.

**iPhone 17, video, 512 MB vs 64 MB cache:**

| Stage | 512 MB cache | 64 MB cache |
|---|---|---|
| videoChunk[0..4] peak | 5199 MB (944 free) | 4769 MB (1374 free) |
| embeds:done | 4879 MB (1264 free) | 4450 MB (**1693 free**) |
| 4-frame result | crash (when hot) | ✅ 28.7 tok/s |
| 8-frame result | ✗ crash | ✅ 25.2 tok/s |
| 16-frame result | ✗ crash | ✅ 27.7 tok/s (when cool) |
| combined result | ✗ crash (×2) | ✅ 28.1 tok/s |

**Throughput did not regress** (27.9 → 29.5 tok/s): the 512 MB cache was hoarding
single-use prefill/vision transients, which don't speed up the decode loop. Fix is
`#if os(iOS)` cap at 64 MB, env-tunable via `GEMMA4_GPU_CACHE_MB`.

## Token-budget capacity model

Convert every input to prefill tokens (clean linear fit from measured `prepare:start tokens=`):

| Media | Tokens |
|---|---|
| Text prompt | actual (~20–45 in tests) |
| Image | **~256** each |
| Video | **~84 / frame** (4f=355, 8f=691, 16f=1363) |
| Audio | ≈ proportional to duration (~200 for test clip) |

**Robustness vs total prefill tokens** (✅ = survives *hot*, i.e. under sustained thermal stress):

| Prefill tokens | Case | Cool | Hot (stress) | Memory warnings (hot) |
|---|---|---|---|---|
| 299 | image | ✅ | ✅ | 1 |
| 691 | video 8f | ✅ | ✅ | 43 (survived) |
| 816 | combined (img + 4f video + audio) | ✅ | ✅ | 22 (survived) |
| 1363 | video 16f | ✅ | ❌ **crash** | 92 (17) / 112 (16) |

**Robust prefill budget ≈ 1000 tokens.** Headroom at `embeds:done` is stable
(~1670–1770 MB); the variable that decides crash-vs-survive is the **prefill forward**,
which scales with total token count. The video vision tower already chunks
(`videoFrameChunkSize=4`) so its peak stays under the ceiling — it is *not* the limiter.

### Thermal degradation
Usable headroom shrinks under sustained inference. 16-frame video passes on a cool phone
and crashes on a warm one (both devices). A shippable gate must keep a margin (~400 MB)
and/or read `os_proc_available_memory()` at request time.

## Recommended gate (production)

```
estimated = textTokens + 256*images + 84*videoFrames + audioTokens(durationSec)
budget    = deriveBudget(os_proc_available_memory())   // ~1000 cool, less when hot
if estimated > budget: trim/degrade (fewer frames / shorter audio / drop an item) + inform user
```
- Conservative caps: video ≤ 8 frames (≈4 s at 2 fps); combined OK; block 16 frames.
- Thermal-adaptive via runtime headroom read.
- UX: trim video to `frameCap / fps` seconds; live "this needs ~N tokens, device supports ~M" messaging; combined handled uniformly by the token math.

## Throughput reference (64 MB cache)
iPhone 17 ≈ 29 tok/s · iPhone 16 ≈ 15 tok/s (identical memory budget; 17 ~2× faster compute).

## Diagnostics added to the fork (profiling only — drop before upstream PR)
- `gemma4MemSnapshot(_:)` — `os_proc_available_memory` + `phys_footprint`, gated by `GEMMA4_MEM_LOG=1`, at each prefill stage.
- `GEMMA4_VIDEO_MAX_FRAMES` — override the iOS frame cap (capped by `config.videoMaxFrames=16`).
- `GEMMA4_LOW_CACHE` — drop the GPU cache right before prefill (found ineffective — the cache must be small at *load* time, not prefill).
- **Trap:** force-evaluating the prefill result before generation materializes the whole
  prefill graph at once, spikes peak memory, and crashes. Rely on MLX's lazy interleave
  with the decode loop instead.
