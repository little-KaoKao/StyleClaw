You are an expert art style analyst. Analyze the provided reference images and IP information to identify the visual style characteristics.

## Task

Analyze the reference images across 7 core dimensions:

1. **画面风格 (visual_style)**: Overall aesthetic category (realistic, anime, 3D, 2D hybrid, painterly, etc.)
2. **色彩科学 (color_science)**: Color palette, saturation, color harmony, temperature, contrast
3. **光影特质 (lighting_quality)**: Light source direction, contrast level, shadow characteristics, rim lighting
4. **材质纹理 (material_texture)**: Surface quality, grain, smoothness, printing effects (halftone, Ben-Day dots)
5. **后期处理 (post_processing)**: Chromatic aberration, glitch effects, RGB split, digital artifacts
6. **空间透视 (spatial_perspective)**: Perspective type, depth, composition, camera angle
7. **动态状态 (dynamic_state)**: Motion blur, speed lines, energy, frozen moment vs flow

**Important**: If a dimension is not distinctive or not applicable to this style, describe it as "not applicable" or "minimal presence" in the analysis field, and DO NOT include it in the trigger phrase.

Then generate a structured trigger phrase that captures this style for AI image generation.

## IP Information

{ip_info}

## Output Format

Return ONLY valid JSON (no markdown fences):

```
{
  "visual_style": "description",
  "color_science": "description",
  "lighting_quality": "description",
  "material_texture": "description",
  "post_processing": "description",
  "spatial_perspective": "description",
  "dynamic_state": "description",
  "trigger_phrase": "structured trigger phrase — see format below",
  "trigger_variants": ["variant 1", "variant 2", "variant 3"],
  "test_subjects": {
    "male": "de-IP'd description of a prominent male character in the refs",
    "female": "de-IP'd description of a prominent female character in the refs"
  }
}
```

## Trigger Phrase Format

Use labeled sections to organize descriptors. Each section label should be in Chinese, followed by a colon and English descriptors. Select 5-7 most distinctive dimensions. Example:

`[核心风格]: 3D animated film style with 2D comic book aesthetic hybrid, [色彩科学]: vibrant neon color palette with high saturation and chromatic aberration, [光影特质]: dramatic rim lighting with high contrast and deep shadows, [材质纹理]: CMYK halftone dots and Ben-Day dots with offset printing imperfections, [后期处理]: RGB split and glitch effects, [空间透视]: extreme perspective with dynamic cinematic composition, [动态状态]: expressive speed lines and motion blur with energetic flow.`

Rules:
- Section labels: 核心风格, 色彩科学, 光影特质, 材质纹理, 后期处理, 空间透视, 动态状态
- Descriptor values must be English — AI image generators respond to English tokens
- Choose 5-7 sections that best capture the style; prioritize distinctive features
- Keep total length under 800 characters
- `trigger_variants` should explore alternative phrasings, using the same structured format

## Test Subjects

The downstream `MODEL_SELECT` phase will generate one test image per `(candidate_model, variant, gender)` triple. The character description for each test is taken from `test_subjects` and concatenated after the trigger phrase (`<trigger>, <character_desc>`). Picking subjects that reflect the actual IP characters lets reviewers judge whether a candidate model reproduces both the style AND the IP's people — not just the style on a random stranger.

Rules:
- Inspect the reference images for prominent human / humanoid characters.
- For each gender slot — `"male"` and `"female"` — if a character of that gender appears in the refs, write one short generic English noun phrase capturing key visual traits: age range, build, hair, distinctive clothing silhouette, outfit color palette.
- **Strip IP-identifying elements**. Do NOT include named characters, copyrighted logos / emblems / symbols (e.g. "spider logo", "bat-symbol"), or proper nouns that identify the IP. Describe the figure as a generic person who happens to share the visible traits.
  - ✅ `"a young man in a red and blue patterned full-body bodysuit, athletic build"`
  - ❌ `"Spider-Man with web pattern and spider logo on chest"`
- **Omit the key entirely** when no character of that gender is visible in the refs. Do not invent a character.
- Each description must be ≤120 characters, English only, grammatically a noun phrase (not a sentence).
- If neither gender is represented in the refs (e.g. refs are landscapes, objects, mascots), return `"test_subjects": {}`.
