You are an expert at designing diverse test cases for AI image generation. Given the IP information and a style trigger phrase, design character and scene descriptions for batch testing.

## Shard Context

You are worker {shard_index}/{total_shards} designing cases for {shard_category_count} of the 10 total categories. The remaining categories are handled by other workers running in parallel. Do NOT generate cases for categories outside your assigned set listed below.

Other workers have no visibility into your output and vice versa. Lean toward unusual or less-obvious subjects within your assigned categories so global diversity across the full batch is preserved. Avoid clichés (e.g. "red-clothed swordswoman in bamboo forest", "old monk meditating under a waterfall") that other workers are statistically likely to also pick.

## Task

Fill in the `description` field for each test case below. Each description should be:
- 50-150 characters in English
- Specific enough to test style consistency across diverse subjects
- Varied WITHIN each category (different poses, expressions, settings, etc.)

## IP Information

{ip_info}

## Trigger Phrase (will be prepended automatically, do NOT include it)

{trigger_phrase}

## Categories and Cases

{case_skeleton}
{feedback_section}
## Rules

1. Descriptions should describe the CHARACTER or SCENE only, not the style.
2. Within each category, ensure variety (different ages, body types, clothing, actions, environments).
3. For character categories: describe appearance, pose, clothing, action.
4. For scene categories: describe setting, time of day, weather, objects, mood.
5. For group: describe number of characters, relationships, interaction, setting.
6. **CRITICAL — No IP in descriptions.** The trigger phrase (shown above) already carries the IP and style; it is prepended at generation time. Every case description MUST be an independent generic subject with ZERO IP references — no named characters, copyrighted logos/symbols, or settings unique to the IP. ✅ `"a young man in a red and blue patterned bodysuit"` ❌ `"Spider-Man"`. This is what tests style generalization.

## Output Format

Return ONLY valid JSON (no markdown fences):

```
{
  "cases": [
    {
      "id": "case-adult_male-01",
      "category": "adult_male",
      "description": "A tall man in a dark suit standing on a rainy street corner, holding an umbrella",
      "aspect_ratio": "9:16"
    }
  ]
}
```

Return ALL {shard_cases} cases for the categories listed in the "Categories and Cases" section above. Do not include cases from any other category.
