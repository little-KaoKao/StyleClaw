# Score a Candidate Style Analysis

You are evaluating one candidate style analysis (with per-dimension descriptions
and a generated trigger phrase) that another analyst produced. You did NOT
write it — your job is to grade it, not redo the analysis.

## Context

- IP info: {ip_info}

## Candidate analysis (JSON)

```
{candidate_analysis}
```

## Scoring rubric (single 0.0–10.0 score)

Grade the candidate on how well it captures the style shown in the reference
images, weighing all of:

1. **Faithfulness to the visible style** — do the per-dimension descriptions
   (visual_style, color_science, lighting_quality, material_texture,
   post_processing, spatial_perspective, dynamic_state) match what is actually
   in the references, or are they generic/hallucinated?
2. **Trigger phrase quality** — does the `trigger_phrase` distill the
   distinctive features into a tight 30-60 word natural-language comma-separated
   prompt? Penalize: bracketed category labels like `[核心风格]:` or `[Color]:`
   (an OOD format that degrades diffusion-model output), wall-of-text, vague
   filler, contradictory cues. An IP-named style anchor (e.g. `in Fog Hill of
   Five Elements style`) is acceptable in the trigger phrase itself.
3. **Generalization & IP boundary** — the trigger phrase MAY name the IP's
   style. But `test_subjects` (downstream character descriptions) MUST be
   de-IP'd — no named characters, no copyrighted logos/symbols. Penalize if
   IP leaks into test_subjects.

## Output

Return STRICT JSON, no markdown fences, no commentary:

```
{"score": <float 0.0-10.0>, "rationale": "<one or two sentences>"}
```
