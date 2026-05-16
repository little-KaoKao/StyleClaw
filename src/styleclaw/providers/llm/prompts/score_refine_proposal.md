# Score a Candidate Trigger Phrase

You are evaluating one candidate trigger phrase that another model produced.
You did NOT write it — your job is to grade it, not rewrite it.

## Context

- IP info: {ip_info}
- Round number: {round_num}
- Recent evaluation history (for context only, do not re-score images):
{history_scores}

## Candidate trigger phrase

```
{candidate_trigger}
```

Optional adjustment note from the author (may be empty):
{candidate_note}

## Scoring rubric (single 0.0–10.0 score)

Grade the candidate on how well it is likely to reproduce the style shown in
the reference images, weighing all of:

1. **Faithfulness to the visible style** (color, line, lighting, texture, mood).
2. **Generalization** — would this phrase work for subjects beyond the IP, or
   has it baked in too much character-specific content?
3. **Concision and clarity** — vague filler, contradictory cues, or wall-of-text
   weakens the score even if every clause is individually reasonable.

## Output

Return STRICT JSON, no markdown fences, no commentary:

```
{"score": <float 0.0-10.0>, "rationale": "<one or two sentences>"}
```
