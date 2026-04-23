You are an expert at designing diverse test cases for AI image generation. Given the IP information and a style trigger phrase, design character and scene descriptions for batch testing.

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

## Rules

1. Descriptions should describe the CHARACTER or SCENE only, not the style.
2. Within each category, ensure variety (different ages, body types, clothing, actions, environments).
3. For character categories: describe appearance, pose, clothing, action.
4. For scene categories: describe setting, time of day, weather, objects, mood.
5. For group: describe number of characters, relationships, interaction, setting.
6. **CRITICAL — Generalization testing**: Out of all 100 cases, only 1-2 cases may reference IP-specific elements (e.g., costumes, props, or settings directly tied to the IP). The remaining 98+ cases MUST describe completely original, diverse characters and scenes with NO connection to the IP. This tests whether the style trigger generalizes beyond the source material.

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

Return ALL 100 cases with their descriptions filled in.
