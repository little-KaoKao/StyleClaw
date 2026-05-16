# Score a Candidate Model Recommendation

You are evaluating one candidate model recommendation (with per-model scores
and a chosen winner) that another evaluator produced. You did NOT write it —
your job is to grade the recommendation, not redo the evaluation.

## Candidate evaluation (JSON)

```
{candidate_evaluation}
```

## Scoring rubric (single 0.0–10.0 score)

Grade the candidate on:

1. **Alignment with the reference images** — does the chosen model/variant in
   fact reproduce the style best in the supplied generations?
2. **Reasonableness of per-model scores** — are the dimension scores defensible
   given what each model produced, or are they obvious miscalls?
3. **Recommendation quality** — is the chosen variant (prompt-only vs
   prompt-sref) the right call given the generations?

## Output

Return STRICT JSON, no markdown fences, no commentary:

```
{"score": <float 0.0-10.0>, "rationale": "<one or two sentences>"}
```
