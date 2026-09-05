# Brightify prompt-builder change: let H3 design the ad

Brief for updating Brightify's MiniMax-H3 integration. It assumes the current
behaviour (four screen references, a time-coded prompt with push-ins,
cross-dissolves, fade-through-white and a generated end card, 544×960, ~10.4 s,
Turbo) and describes what to replace it with.

## Why

MiniMax-H3 with the Turbo LoRA (6–7 denoising passes) is stable when it chooses
the shot design itself and unstable when the prompt dictates an edit. Measured
on production output: wobble and colour flutter concentrate in the "camera
pushes in" segments, the ghosting and botched cut are the cross-dissolves, and
the only stable segment is the static end card. The same references with the
prompt "create an awesome vertical tiktok ad of these screens" produce a clean
clip. So the prompt builder must stop writing an edit decision list and start
writing a creative brief.

## What the prompt must contain (in this order)

1. **Reference roles, one line per image.** H3 addresses references as
   `<Picture 1>`, `<Picture 2>`, … in upload order, so the order of
   `reference_image_urls` must match the order the prompt describes. Each line
   says what the image is and that it must stay legible and unchanged.
2. **The ad brief.** Product name, what it does in one clause, target viewer,
   mood/energy. Two or three sentences of natural prose.
3. **Format and pacing at the level of intent.** "Vertical 9:16 TikTok ad",
   "energetic quick cuts", "each screen shown full-frame, sharp and readable",
   "show the screens in order". No timestamps, no per-shot camera moves.
4. **Optional audio line, only if `include_audio` is true:** a final line
   starting with `overall_soundscape:` describing music/SFX. Omit entirely when
   audio is off.

Target 60–140 words total, plain sentences, no brackets, no bullet lists inside
the prompt.

## Template

```text
<Picture 1> is the {screen 1 name} screen of {Product}. <Picture 2> is the {screen 2 name} screen. <Picture 3> is the {screen 3 name} screen. <Picture 4> is the {screen 4 name} screen. Keep every screen exactly as shown: same layout, text, colours and proportions, always sharp and readable.

An energetic vertical 9:16 TikTok ad for {Product}, {one-clause value proposition}, aimed at {audience}. {Mood: e.g. upbeat, modern, confident}. Quick snap cuts from one screen to the next, in order, each screen filling the frame on a clean {background description} background with lively UI motion inside the screen. Fast, punchy, polished.

overall_soundscape: {only when include_audio is true — e.g. upbeat electronic track with crisp UI tap sounds}
```

## Vocabulary rules

Allowed motion words: *quick cuts, snap cuts, hard cut, UI elements animate,
buttons tap, content scrolls, screen slides in, pops in, locked-off, static*.

Forbidden (each one is a measured failure mode):

- **Transitions:** cross-dissolve, dissolve, fade, fade through white/black,
  crossfade, morph, blend into.
- **Slow camera moves on flat plates:** push in, dolly in, slow zoom, Ken
  Burns, drift, handheld, orbit, parallax.
- **Time codes and shot lists:** `[0-3s]`, "at 4 seconds", "shot 1 / shot 2".
- **Negative instructions:** "no camera shake", "no flicker", "no distortion",
  "no ghosting". Turbo runs at CFG 1 so there is no negative branch; naming an
  artifact only adds the concept to the conditioning.
- **Rendered text the model must invent:** taglines, prices, CTAs, logos.
  Everything textual must already exist inside a reference screen.

## End card

Do not ask H3 to generate the end card. Two options, in order of preference:

- **Composite in post.** Generate only the screens portion, then overlay or
  append the real end-card PNG for the final ~1.5 s with the video toolchain
  Brightify already uses for resizing. Pixel-perfect logo, zero model risk.
- **If it must be generated,** upload the end card as the last reference and
  add one sentence: "<Picture 5> is the closing card; it fills the frame at the
  very end and holds still." Accept that logo text may soften.

## Request parameters

```json
{
  "prompt": "<built as above>",
  "reference_image_urls": ["<screen1>", "<screen2>", "<screen3>", "<screen4>"],
  "width": 544,
  "height": 960,
  "seconds": 10.4,
  "turbo": true,
  "turbo_strength": 1.0,
  "include_audio": false,
  "output_width": 1080,
  "output_height": 1920,
  "seed": -1
}
```

Notes for the implementer:

- `width`/`height` must be multiples of 32; 544×960 is correct for 9:16 and
  matches the Turbo LoRA's 544p training resolution. Keep upscaling to
  1080×1920 via `output_width`/`output_height` (bilinear, antialiased, no
  retiming).
- `seconds` is snapped **down** to the 17n+5 frame grid at 24 fps. 10.4 s
  becomes 243 frames = 10.125 s. Valid lengths: 5.17, 5.88, 6.58, 7.29, 8.0,
  8.71, 9.42, 10.13, 10.83, 11.54, 12.25, 12.96, 13.67, 14.38 s. Show the user
  the snapped value rather than the requested one. Only set
  `output_duration_seconds` if an exact length is mandatory; it pads by
  freezing the last frame.
- Leave `steps` unset (defaults to 7 with Turbo) and `turbo_strength` at 1.0.
  Expose a "final quality" toggle that sends `"turbo": false` (~50 steps,
  ~5–6× slower) for the hero render once the prompt is approved.
- Use a fixed `seed` while iterating on a prompt so changes are attributable;
  `-1` for production variety.
- Reference images: 1–9 allowed, use 3–5. Supply them at or above 544×960
  pixels (the loader never upscales), as HTTP(S) URLs, in the same order the
  prompt names them. Screenshots should be clean full-screen captures, not
  device mock-ups with shadows, unless the mock-up is the intended look.
- `include_audio: false` unless audio is wanted; if true, add the
  `overall_soundscape:` line.

## If Brightify uses an LLM to write the prompt

Give the rewriter this system instruction:

> Turn the product info and screen list into a MiniMax-H3 prompt following the
> template exactly. Describe the ad's intent, mood and pacing; never write
> timestamps, shot lists, dissolves, fades, zooms, push-ins, camera moves,
> negative instructions, or any on-screen text that is not already in a
> reference. Name each `<Picture n>` once and require it to stay unchanged and
> readable.

Validate the output with a regex reject-list for the forbidden words before
submitting.

## Acceptance test

For one real product, submit two jobs with identical references, seed and
parameters: the old time-coded prompt and the new brief-style prompt. The new
one should show no double-exposed frames, no shimmer on screen edges during the
whole clip, and each screen readable while on screen. Then run the same brief
with `"turbo": false` once to confirm the full-quality path is at least as good.
