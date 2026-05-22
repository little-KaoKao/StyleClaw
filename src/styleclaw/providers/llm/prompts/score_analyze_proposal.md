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
   distinctive features into a tight, structured prompt that a downstream
   image generator can act on? Vague filler, contradictory cues, or
   wall-of-text weakens the score.
3. **Generalization** — would the trigger work for subjects beyond the IP, or
   has it baked in too much character-specific content?

## Output

Return STRICT JSON, no markdown fences, no commentary:

```
{"score": <float 0.0-10.0>, "rationale": "<one or two sentences>"}
```
