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
3. Age-category contract is mandatory:
   - `adult_male`, `adult_female`: 20-30 years old only. Prefer explicit phrasing like `in his 20s`, `in her late 20s`, or `age 24`. Do not use 30s/40s/50s, middle-aged, senior, elderly, silver-streaked, gray-haired, grey-haired, or salt-and-pepper descriptors.
   - `little_male_child`, `little_female_child`: 8-14 years old only. Prefer explicit phrasing like `age 8`, `age 12`, or `14-year-old`. Do not use baby, infant, toddler, preschooler, kindergarten-age, or under-8 child descriptors.
   - `elderly_male`, `elderly_female`: 50+ years old. Prefer explicit phrasing like `in his 50s`, `in her 60s`, or `age 70`.
4. For character categories: describe appearance, pose, clothing, action. Default to natural, neutral, relaxed, focused, confident, warm, or lightly upbeat expressions. Do NOT infer melodrama from broad genre words like "drama". Avoid high-distress cues unless the user feedback or IP information explicitly asks for them: tears, crying, red eyes, worried, anxious, nervous, tense, angry, regret, sadness, grief, arguments, divorce, breakup, custody disputes, police emergencies, hospital crises, funerals, blackmail, secrets, or confrontation.
5. For scene categories (`outdoor_scene`, `indoor_scene`): describe setting, time of day, weather, objects, mood. These MUST be empty environment shots with an explicit no-people constraint: include `no people, no human figures, no silhouettes, no portraits`. Do not mention crowds, commuters, guests, staff, neighbors, reporters, reflections of people, or any visible character.
6. For group: describe number of characters, relationships, interaction, setting. Keep interaction readable but not melodramatic by default; avoid fights, crying, grief, courtroom conflict, hospital crisis, or breakup scenes unless explicitly requested.
7. **CRITICAL — No IP in descriptions.** The trigger phrase (shown above) already carries the IP and style; it is prepended at generation time. Every case description MUST be an independent generic subject with ZERO IP references — no named characters, copyrighted logos/symbols, or settings unique to the IP. ✅ `"a young man in a red and blue patterned bodysuit"` ❌ `"Spider-Man"`. This is what tests style generalization.

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
