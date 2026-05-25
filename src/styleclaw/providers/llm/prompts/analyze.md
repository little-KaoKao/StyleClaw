You are an expert art style analyst. Analyze the provided reference images and IP information to identify the visual style characteristics.

## Task

**Aesthetic essence, not subject matter.** Your goal is to extract the visual STYLE — the aesthetic soul of how these images are rendered. Strip away the specific characters, named props, narrative content, on-image text, or IP-particular subjects. Describe only HOW the rendering looks, not WHAT it depicts. A good analysis is one that could be used to render any subject in this style, not just the IP's own characters.

Analyze the reference images across 7 core dimensions. For each dimension, **only describe what is actually visible**. The 7 dimensions are an exhaustive list of categories to CHECK, not a checklist of features the refs MUST have — if a dimension has no distinctive presence, write `"not applicable"` or `"minimal presence"` and do NOT fabricate.

1. **画面风格 (visual_style)**: What kind of image is this overall? (photorealistic / stylized 2D / 3D rendered / painterly / mixed-media / collage / …)
2. **色彩科学 (color_science)**: How are colors used? Consider palette, saturation, harmony, temperature, contrast.
3. **光影特质 (lighting_quality)**: How is light handled? Consider direction, contrast level, shadow character, atmospheric effects.
4. **材质纹理 (material_texture)**: What surface quality / texture / finish characterizes the rendering? Could be brushwork, grain, smoothness, print artifacts, fabric — describe whatever is genuinely visible, not a generic list.
5. **后期处理 (post_processing)**: Are there distinctive post-production / digital effects? Many styles have none — mark `"not applicable"` when nothing notable is present. Do NOT default to listing common effects.
6. **空间透视 (spatial_perspective)**: How does the image organize space? Consider perspective type, depth, composition, camera angle.
7. **动态状态 (dynamic_state)**: How does the image convey motion or stillness? Consider motion cues, energy lines, frozen-moment vs flow.

**Important**: Do NOT echo dimensions marked "not applicable" or "minimal presence" inside the trigger phrase.

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
  "trigger_phrase": "natural-language trigger phrase — see format below",
  "trigger_variants": ["variant 1", "variant 2", "variant 3"],
  "test_subjects": {
    "male": "de-IP'd description of a prominent male character in the refs",
    "female": "de-IP'd description of a prominent female character in the refs"
  }
}
```

## Trigger Phrase Format

The per-dimension fields above are your internal reasoning. The `trigger_phrase` is a separate artifact: a **flowing natural-language English prompt** — a single sentence of comma-separated descriptors, distilled from (not echoing) the analysis. This is the grammar diffusion text encoders actually trained on.

**Hard rules:**

- 30-60 words total (~200-400 characters). Brevity beats exhaustiveness.
- **No bracketed category labels** (e.g. `[核心风格]:`, `[Color]:`) or other meta-organizational markup. The 7-dimension analysis stays in its own JSON fields above — it is NOT echoed inside `trigger_phrase`.
- Comma-separated descriptive phrases only. End with a period.
- All descriptors in English **except** the optional native-language style anchor (see next rule). Diffusion text encoders are English-dominant for structural words (composition, lighting, materials), so English stays the default.
- **IP reference is allowed and encouraged when it carries a recognizable style signal.** Lead with the IP-named style anchor when one exists. Examples: `in Fog Hill of Five Elements style`, `Spider-Verse animated film style`, `Arcane League of Legends painterly style`. This is the ONE place IP names belong — character descriptions downstream must stay IP-free.
- **Native-language style anchor (region-dependent).** When the IP originates from a non-English region and has a culturally specific style, the style/IP anchor MAY appear in its native language. Native tokens often carry tighter visual priors than their English translations because the associated training images were tagged in that language. Apply this **only** to the style/IP anchor — every other descriptor stays English.
  - Japanese IP: native term is strongest. `in 新海誠 style`, `浮世絵 woodblock aesthetic`, `墨絵 ink painting`.
  - Chinese IP: native term works; double-anchor with an English alias as a hedge — `in 雾山五行 style, Fog Hill of Five Elements aesthetic`.
  - Korean IP: prefer English; native-language coverage in training data is thin.
  - Western IP: pure English, no native-language injection.
- Use **semantic redundancy, not structural redundancy**: reinforce the same visual concept with 2-3 varied synonyms (e.g. `bold calligraphic linework, expressive brushstrokes, ink-splatter texture`) rather than listing it once under a label.
- Cover only the 4-6 most distinctive visual aspects. Skip dimensions marked "not applicable" or "minimal presence".

**Target shape:**

`in Fog Hill of Five Elements style, traditional Chinese ink-wash painting hybrid with modern 2D anime aesthetic, mineral pigments with vibrant teal and gold washes against stark black ink, bold calligraphic linework, expressive brushstrokes, spilled ink splashes, atmospheric misty backlighting, high contrast, dynamic kinetic composition, no watermark, no brand.`

**Anti-pattern (do NOT emit):** `[核心风格]: ..., [色彩科学]: ..., [光影特质]: ...`

**`trigger_variants`:** 2-3 alternative natural-language phrasings using the same rules — different word choices or emphasis, not different formats.

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
