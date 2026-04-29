# StyleClaw: Extended Thinking Display + Bidirectional Phase Transitions

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (A) Capture and expose Claude's extended thinking output during LLM agent calls so users can see the reasoning process. (B) Loosen the phase state machine so MODEL_SELECT can be re-entered from STYLE_REFINE / BATCH_T2I, and BATCH_I2I can return to BATCH_T2I — enabling a workflow of "refine trigger → re-test all models → pick best → batch test → if batch fails, go back and adjust".

**Architecture:** Part A adds an optional `thinking` channel to `LLMProvider.invoke()` returning a `(text, thinking)` tuple when requested; BedrockProvider requests Claude extended thinking via the `thinking` block and separates `thinking` / `text` content blocks in the response. Agents that do heavy reasoning (analyze_style, select_model, refine_prompt, evaluate_result) opt in, and their output is persisted beside the JSON result as `*.thinking.md`. CLI gets a `--show-thinking` flag that streams the thinking to stdout. Part B extends `TRANSITIONS` to allow the new edges, versions the `model-select/` directory with pass folders (`pass-001/`, `pass-002/`) so re-entry doesn't clobber prior data, and adapts scripts/storage/reports to read from the *current* pass.

**Tech Stack:** Python 3.11+, Pydantic v2, httpx, Typer, pytest (+ respx for HTTP mocking). Follow existing frozen-model + `with_*` builder patterns.

---

## Part A: Extended Thinking Display

### File Structure (Part A)

**Modify:**
- [src/styleclaw/providers/llm/base.py](src/styleclaw/providers/llm/base.py) — extend Protocol with optional `thinking_budget` and return type
- [src/styleclaw/providers/llm/bedrock.py](src/styleclaw/providers/llm/bedrock.py) — request thinking block, parse response, return both channels
- [src/styleclaw/core/text_utils.py](src/styleclaw/core/text_utils.py) — no change, just noting it stays as-is
- [src/styleclaw/agents/analyze_style.py](src/styleclaw/agents/analyze_style.py) — accept optional `thinking_budget`, return thinking alongside result
- [src/styleclaw/agents/select_model.py](src/styleclaw/agents/select_model.py) — same pattern
- [src/styleclaw/agents/refine_prompt.py](src/styleclaw/agents/refine_prompt.py) — same pattern
- [src/styleclaw/agents/evaluate_result.py](src/styleclaw/agents/evaluate_result.py) — same pattern
- [src/styleclaw/orchestrator/actions.py](src/styleclaw/orchestrator/actions.py) — read flag from ctx, forward to agents, save thinking to disk
- [src/styleclaw/orchestrator/actions.py](src/styleclaw/orchestrator/actions.py) — `ExecutionContext` gains `show_thinking: bool` and `thinking_budget: int`
- [src/styleclaw/cli.py](src/styleclaw/cli.py) — add `--show-thinking` flag to analyze / evaluate / refine / run; print thinking to stdout
- [src/styleclaw/storage/project_store.py](src/styleclaw/storage/project_store.py) — add `save_thinking()` helper

**Create:**
- `tests/providers/llm/test_bedrock_thinking.py`
- `tests/agents/test_thinking_passthrough.py`
- `tests/orchestrator/test_actions_thinking.py`

---

### Task A1: Extend LLMProvider Protocol to support thinking

**Files:**
- Modify: [src/styleclaw/providers/llm/base.py](src/styleclaw/providers/llm/base.py)

The Protocol currently returns `str`. We need a backward-compatible extension so callers who don't care about thinking see no change, and callers who opt in get both channels.

Approach: keep `invoke()` returning `str` for existing code paths; add a new method `invoke_with_thinking()` that returns `LLMResponse` (a small dataclass). This avoids breaking every existing call site.

- [ ] **Step 1: Write the failing test**

Create `tests/providers/llm/test_base.py` additions — actually just add a new test file since `test_base.py` already exists. Append to existing file:

```python
# Add to tests/providers/llm/test_base.py

from styleclaw.providers.llm.base import LLMProvider, LLMResponse


class TestLLMResponse:
    def test_has_text_and_thinking_fields(self):
        r = LLMResponse(text="hi", thinking="because")
        assert r.text == "hi"
        assert r.thinking == "because"

    def test_thinking_defaults_to_empty(self):
        r = LLMResponse(text="hi")
        assert r.thinking == ""


class TestLLMProviderProtocol:
    def test_protocol_requires_invoke(self):
        class Fake:
            async def invoke(self, system, messages, max_tokens=4096, temperature=0.3):
                return "ok"
        assert isinstance(Fake(), LLMProvider)

    def test_protocol_requires_invoke_with_thinking(self):
        class Fake:
            async def invoke(self, system, messages, max_tokens=4096, temperature=0.3):
                return "ok"
            async def invoke_with_thinking(self, system, messages, max_tokens=4096, thinking_budget=5000):
                return LLMResponse(text="ok", thinking="why")
        assert isinstance(Fake(), LLMProvider)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/providers/llm/test_base.py -v`
Expected: FAIL — `LLMResponse` does not exist, `LLMProvider` has no `invoke_with_thinking`.

- [ ] **Step 3: Write minimal implementation**

Replace the full contents of [src/styleclaw/providers/llm/base.py](src/styleclaw/providers/llm/base.py):

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMResponse:
    text: str
    thinking: str = ""


@runtime_checkable
class LLMProvider(Protocol):
    async def invoke(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str: ...

    async def invoke_with_thinking(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        thinking_budget: int = 5000,
    ) -> LLMResponse: ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/providers/llm/test_base.py -v`
Expected: PASS for all 4 new tests plus any existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/providers/llm/base.py tests/providers/llm/test_base.py
git commit -m "feat(llm): add LLMResponse and invoke_with_thinking protocol method"
```

---

### Task A2: Implement thinking in BedrockProvider

**Files:**
- Modify: [src/styleclaw/providers/llm/bedrock.py](src/styleclaw/providers/llm/bedrock.py)
- Create: `tests/providers/llm/test_bedrock_thinking.py`

Bedrock/Anthropic extended thinking: request body gets a `"thinking": {"type": "enabled", "budget_tokens": N}` field; response `content` array includes blocks with `type == "thinking"` (field `thinking`, containing the reasoning text) alongside `type == "text"` blocks. Temperature MUST be 1.0 when thinking is enabled (Claude hard constraint).

- [ ] **Step 1: Write the failing test**

Create `tests/providers/llm/test_bedrock_thinking.py`:

```python
from __future__ import annotations

import json

import httpx
import pytest
import respx

from styleclaw.providers.llm.base import LLMResponse
from styleclaw.providers.llm.bedrock import BedrockProvider


@pytest.fixture
def provider(monkeypatch) -> BedrockProvider:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-token")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    return BedrockProvider(region="us-east-1", model_id="test-model")


class TestBedrockInvokeWithThinking:
    @respx.mock
    async def test_returns_both_text_and_thinking(self, provider: BedrockProvider) -> None:
        route = respx.post(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        ).respond(
            json={
                "content": [
                    {"type": "thinking", "thinking": "Let me reason about this..."},
                    {"type": "text", "text": "final answer"},
                ],
            }
        )

        result = await provider.invoke_with_thinking(
            system="s",
            messages=[{"role": "user", "content": "q"}],
            thinking_budget=5000,
        )
        assert isinstance(result, LLMResponse)
        assert result.text == "final answer"
        assert result.thinking == "Let me reason about this..."
        assert route.called

    @respx.mock
    async def test_request_body_enables_thinking(self, provider: BedrockProvider) -> None:
        captured = {}

        def _handler(request):
            captured["body"] = json.loads(request.content.decode())
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "ok"}]},
            )

        respx.post(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        ).mock(side_effect=_handler)

        await provider.invoke_with_thinking(
            system="s", messages=[], thinking_budget=3000,
        )
        assert captured["body"]["thinking"] == {"type": "enabled", "budget_tokens": 3000}
        # Extended thinking requires temperature == 1.0
        assert captured["body"]["temperature"] == 1.0

    @respx.mock
    async def test_thinking_empty_when_no_thinking_block(
        self, provider: BedrockProvider,
    ) -> None:
        respx.post(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        ).respond(json={"content": [{"type": "text", "text": "answer"}]})

        result = await provider.invoke_with_thinking(
            system="s", messages=[], thinking_budget=5000,
        )
        assert result.text == "answer"
        assert result.thinking == ""

    @respx.mock
    async def test_joins_multiple_thinking_blocks(
        self, provider: BedrockProvider,
    ) -> None:
        respx.post(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        ).respond(
            json={
                "content": [
                    {"type": "thinking", "thinking": "step 1"},
                    {"type": "thinking", "thinking": "step 2"},
                    {"type": "text", "text": "done"},
                ],
            }
        )
        result = await provider.invoke_with_thinking(
            system="s", messages=[], thinking_budget=5000,
        )
        assert result.thinking == "step 1\n\nstep 2"
        assert result.text == "done"

    @respx.mock
    async def test_invoke_without_thinking_unchanged(
        self, provider: BedrockProvider,
    ) -> None:
        """Existing invoke() must not regress."""
        captured = {}

        def _handler(request):
            captured["body"] = json.loads(request.content.decode())
            return httpx.Response(
                200, json={"content": [{"type": "text", "text": "ok"}]},
            )

        respx.post(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        ).mock(side_effect=_handler)

        result = await provider.invoke(system="s", messages=[], temperature=0.3)
        assert result == "ok"
        assert "thinking" not in captured["body"]
        assert captured["body"]["temperature"] == 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/providers/llm/test_bedrock_thinking.py -v`
Expected: FAIL — `invoke_with_thinking` not defined on BedrockProvider.

- [ ] **Step 3: Write the implementation**

Replace the full contents of [src/styleclaw/providers/llm/bedrock.py](src/styleclaw/providers/llm/bedrock.py):

```python
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Self

import httpx

from styleclaw.providers.llm.base import LLMResponse

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


class BedrockProvider:
    def __init__(
        self,
        region: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self._region = region or os.getenv("AWS_REGION", "")
        if not self._region:
            self._region = "us-east-1"
            logger.warning("AWS_REGION not set, defaulting to 'us-east-1'")
        self._model_id = model_id or os.getenv(
            "CLAUDE_MODEL", "anthropic.claude-sonnet-4-20250514"
        )
        bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")
        if not bearer_token:
            raise ValueError(
                "AWS_BEARER_TOKEN_BEDROCK is not set. "
                "Please set it in your .env file or environment."
            )
        base_url = f"https://bedrock-runtime.{self._region}.amazonaws.com"
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json",
            },
            timeout=120,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    async def invoke(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        result = await self._post(body)
        text_blocks = [
            b["text"] for b in result.get("content", [])
            if b.get("type") == "text"
        ]
        if not text_blocks:
            raise ValueError("Bedrock returned no text content in response")
        return "\n".join(text_blocks)

    async def invoke_with_thinking(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        thinking_budget: int = 5000,
    ) -> LLMResponse:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            # Extended thinking requires temperature == 1.0.
            "temperature": 1.0,
            "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
        }
        result = await self._post(body)
        blocks = result.get("content", [])
        text_parts = [b["text"] for b in blocks if b.get("type") == "text"]
        thinking_parts = [
            b.get("thinking", "") for b in blocks if b.get("type") == "thinking"
        ]
        if not text_parts:
            raise ValueError("Bedrock returned no text content in response")
        return LLMResponse(
            text="\n".join(text_parts),
            thinking="\n\n".join(t for t in thinking_parts if t),
        )

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        url = f"/model/{self._model_id}/invoke"
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = await self._http.post(url, content=json.dumps(body))
                resp.raise_for_status()
                return resp.json()
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    wait = 2**attempt
                    logger.warning(
                        "Bedrock request failed (attempt %d/%d): %s. Retrying in %ds.",
                        attempt + 1, MAX_RETRIES, exc, wait,
                    )
                    await asyncio.sleep(wait)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    wait = 2**attempt
                    logger.warning(
                        "Bedrock request failed (attempt %d/%d): %s. Retrying in %ds.",
                        attempt + 1, MAX_RETRIES, exc, wait,
                    )
                    await asyncio.sleep(wait)
        raise RuntimeError(
            f"Bedrock invoke failed after {MAX_RETRIES} retries"
        ) from last_exc
```

- [ ] **Step 4: Run new tests**

Run: `uv run python -m pytest tests/providers/llm/test_bedrock_thinking.py -v`
Expected: PASS for all 5 tests.

- [ ] **Step 5: Run the existing Bedrock tests to check no regression**

Run: `uv run python -m pytest tests/providers/llm/test_bedrock.py -v`
Expected: PASS — all existing tests still pass (we preserved `invoke()` shape and `_post()` was refactored but covers the same retry behavior).

- [ ] **Step 6: Commit**

```bash
git add src/styleclaw/providers/llm/bedrock.py tests/providers/llm/test_bedrock_thinking.py
git commit -m "feat(llm): BedrockProvider.invoke_with_thinking surfaces extended thinking"
```

---

### Task A3: Persist thinking alongside agent outputs (storage helper)

**Files:**
- Modify: [src/styleclaw/storage/project_store.py](src/styleclaw/storage/project_store.py)

Add a single helper that writes a `.thinking.md` file next to any output JSON. Agents will call it with the absolute path of the JSON file they just saved.

- [ ] **Step 1: Write the failing test**

Append to `tests/storage/test_project_store.py`:

```python
class TestSaveThinking:
    def test_writes_thinking_md_next_to_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")
        target = tmp_path / "projects" / "p" / "analysis.json"
        target.parent.mkdir(parents=True)
        target.write_text("{}")

        project_store.save_thinking(target, "I reasoned step-by-step.")

        md = target.with_suffix(".thinking.md")
        assert md.exists()
        assert "I reasoned step-by-step." in md.read_text(encoding="utf-8")

    def test_empty_thinking_does_not_write_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")
        target = tmp_path / "projects" / "p" / "analysis.json"
        target.parent.mkdir(parents=True)

        project_store.save_thinking(target, "")

        md = target.with_suffix(".thinking.md")
        assert not md.exists()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run python -m pytest tests/storage/test_project_store.py::TestSaveThinking -v`
Expected: FAIL — `save_thinking` not defined.

- [ ] **Step 3: Implement**

Append to [src/styleclaw/storage/project_store.py](src/styleclaw/storage/project_store.py) (before the `_read_json` helper at the end of the file):

```python
def save_thinking(json_path: Path, thinking: str) -> None:
    """Write thinking text to a sibling .thinking.md file.

    No-op when thinking is empty. Called by agents after saving their JSON output.
    """
    if not thinking:
        return
    md_path = json_path.with_suffix(".thinking.md")
    md_path.write_text(thinking, encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run python -m pytest tests/storage/test_project_store.py::TestSaveThinking -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/storage/project_store.py tests/storage/test_project_store.py
git commit -m "feat(storage): add save_thinking() helper for .thinking.md sidecars"
```

---

### Task A4: Thread thinking through the analyze_style agent

**Files:**
- Modify: [src/styleclaw/agents/analyze_style.py](src/styleclaw/agents/analyze_style.py)

Pattern (applies to A4–A7): keep the old `analyze_style(...)` returning `StyleAnalysis`. Add a new `analyze_style_with_thinking(...)` returning `(StyleAnalysis, str)`. The non-thinking function just delegates and discards thinking. Callers (actions.py) call the `_with_thinking` variant when the flag is on.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_thinking_passthrough.py`:

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from styleclaw.agents.analyze_style import (
    analyze_style,
    analyze_style_with_thinking,
)
from styleclaw.core.models import StyleAnalysis
from styleclaw.providers.llm.base import LLMResponse


@pytest.fixture
def ref_image(tmp_path: Path) -> Path:
    from PIL import Image
    p = tmp_path / "ref.png"
    Image.new("RGB", (32, 32), "red").save(p)
    return p


class TestAnalyzeStyleWithThinking:
    async def test_returns_analysis_and_thinking(self, ref_image, tmp_path):
        fake_llm = AsyncMock()
        fake_llm.invoke_with_thinking = AsyncMock(
            return_value=LLMResponse(
                text='{"trigger_phrase": "bold ink style"}',
                thinking="I observed heavy linework and high contrast.",
            )
        )

        analysis, thinking = await analyze_style_with_thinking(
            fake_llm, [ref_image], ip_info="test ip", thinking_budget=3000,
        )
        assert isinstance(analysis, StyleAnalysis)
        assert analysis.trigger_phrase == "bold ink style"
        assert thinking == "I observed heavy linework and high contrast."

    async def test_plain_analyze_style_still_returns_just_analysis(
        self, ref_image,
    ):
        fake_llm = AsyncMock()
        fake_llm.invoke = AsyncMock(
            return_value='{"trigger_phrase": "plain"}'
        )
        analysis = await analyze_style(fake_llm, [ref_image], ip_info="test")
        assert isinstance(analysis, StyleAnalysis)
        assert analysis.trigger_phrase == "plain"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run python -m pytest tests/agents/test_thinking_passthrough.py::TestAnalyzeStyleWithThinking -v`
Expected: FAIL — `analyze_style_with_thinking` not importable.

- [ ] **Step 3: Modify the agent**

Replace the body of [src/styleclaw/agents/analyze_style.py](src/styleclaw/agents/analyze_style.py) with:

```python
from __future__ import annotations

import logging
from pathlib import Path

from styleclaw.core.image_utils import build_image_block
from styleclaw.core.models import StyleAnalysis
from styleclaw.core.text_utils import parse_llm_response, sanitize_braces
from styleclaw.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent / "providers" / "llm" / "prompts" / "analyze.md"


def _build_messages(ref_image_paths: list[Path]) -> list[dict]:
    content: list[dict] = [build_image_block(p) for p in ref_image_paths]
    content.append({
        "type": "text",
        "text": "Analyze these reference images and generate a style trigger phrase.",
    })
    return [{"role": "user", "content": content}]


def _build_system_prompt(ip_info: str) -> str:
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.replace("{ip_info}", sanitize_braces(ip_info))


async def analyze_style(
    llm: LLMProvider,
    ref_image_paths: list[Path],
    ip_info: str,
) -> StyleAnalysis:
    raw = await llm.invoke(
        system=_build_system_prompt(ip_info),
        messages=_build_messages(ref_image_paths),
    )
    analysis = parse_llm_response(raw, StyleAnalysis, "style analysis")
    logger.info("Style analysis complete. Trigger: %s", analysis.trigger_phrase[:80])
    return analysis


async def analyze_style_with_thinking(
    llm: LLMProvider,
    ref_image_paths: list[Path],
    ip_info: str,
    thinking_budget: int = 5000,
) -> tuple[StyleAnalysis, str]:
    response = await llm.invoke_with_thinking(
        system=_build_system_prompt(ip_info),
        messages=_build_messages(ref_image_paths),
        thinking_budget=thinking_budget,
    )
    analysis = parse_llm_response(response.text, StyleAnalysis, "style analysis")
    logger.info("Style analysis (with thinking) complete. Trigger: %s", analysis.trigger_phrase[:80])
    return analysis, response.thinking
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run python -m pytest tests/agents/test_thinking_passthrough.py tests/agents/test_analyze_style.py -v`
Expected: PASS — new thinking tests and existing tests all green.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/agents/analyze_style.py tests/agents/test_thinking_passthrough.py
git commit -m "feat(agents): analyze_style_with_thinking returns reasoning alongside result"
```

---

### Task A5: Thread thinking through refine_prompt agent

**Files:**
- Modify: [src/styleclaw/agents/refine_prompt.py](src/styleclaw/agents/refine_prompt.py)
- Modify: `tests/agents/test_thinking_passthrough.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/agents/test_thinking_passthrough.py`:

```python
from styleclaw.agents.refine_prompt import (
    refine_prompt,
    refine_prompt_with_thinking,
)
from styleclaw.core.models import PromptConfig


class TestRefinePromptWithThinking:
    async def test_returns_config_and_thinking(self, ref_image):
        fake_llm = AsyncMock()
        fake_llm.invoke_with_thinking = AsyncMock(
            return_value=LLMResponse(
                text='{"trigger_phrase": "new trigger"}',
                thinking="Color scored low so I added 'vivid palette'.",
            )
        )
        config, thinking = await refine_prompt_with_thinking(
            fake_llm, [ref_image],
            current_trigger="old trigger", round_num=2,
            ip_info="ip", evaluations=[], human_direction="",
            thinking_budget=3000,
        )
        assert isinstance(config, PromptConfig)
        assert config.trigger_phrase == "new trigger"
        assert config.round == 2
        assert thinking.startswith("Color scored low")
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run python -m pytest tests/agents/test_thinking_passthrough.py::TestRefinePromptWithThinking -v`
Expected: FAIL — `refine_prompt_with_thinking` not importable.

- [ ] **Step 3: Modify the agent**

Replace the body of [src/styleclaw/agents/refine_prompt.py](src/styleclaw/agents/refine_prompt.py) with:

```python
from __future__ import annotations

import json
import logging
from pathlib import Path

from styleclaw.core.image_utils import build_image_block
from styleclaw.core.models import PromptConfig, RoundEvaluation
from styleclaw.core.text_utils import clean_json, parse_llm_response, sanitize_braces
from styleclaw.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "providers" / "llm" / "prompts" / "refine.md"
)

MAX_HISTORY_ROUNDS = 3


def _build_history_text(evaluations: list[RoundEvaluation]) -> str:
    if not evaluations:
        return "(no previous evaluations)"
    recent = evaluations[-MAX_HISTORY_ROUNDS:]
    lines: list[str] = []
    for ev in recent:
        lines.append(f"### Round {ev.round}")
        for score in ev.evaluations:
            s = score.scores
            lines.append(
                f"- {score.model}: color={s.color_palette} line={s.line_style} "
                f"light={s.lighting} texture={s.texture} mood={s.overall_mood} "
                f"total={score.total:.1f}"
            )
        if ev.next_direction:
            lines.append(f"  Direction: {ev.next_direction}")
    return "\n".join(lines)


def _build_system_prompt(
    current_trigger: str,
    round_num: int,
    ip_info: str,
    evaluations: list[RoundEvaluation],
    human_direction: str,
) -> str:
    history_text = _build_history_text(evaluations)
    return (
        PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("{trigger_phrase}", current_trigger)
        .replace("{round_num}", str(round_num))
        .replace("{ip_info}", sanitize_braces(ip_info))
        .replace("{history_scores}", history_text)
        .replace("{human_direction}", sanitize_braces(human_direction) if human_direction else "(none)")
    )


def _build_messages(ref_image_paths: list[Path]) -> list[dict]:
    content: list[dict] = [build_image_block(p) for p in ref_image_paths]
    content.append({
        "type": "text",
        "text": "Refine the trigger phrase based on the evaluation history and reference images.",
    })
    return [{"role": "user", "content": content}]


def _parse_prompt_config(raw_text: str, round_num: int) -> PromptConfig:
    cleaned = clean_json(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON for prompt refinement: {exc}") from exc
    data["round"] = round_num
    data.setdefault(
        "derived_from",
        f"round-{round_num - 1:03d}" if round_num > 1 else "initial-analysis",
    )
    return parse_llm_response(
        json.dumps(data), PromptConfig, "prompt refinement",
    )


async def refine_prompt(
    llm: LLMProvider,
    ref_image_paths: list[Path],
    current_trigger: str,
    round_num: int,
    ip_info: str,
    evaluations: list[RoundEvaluation],
    human_direction: str = "",
) -> PromptConfig:
    system_prompt = _build_system_prompt(
        current_trigger, round_num, ip_info, evaluations, human_direction,
    )
    raw = await llm.invoke(system=system_prompt, messages=_build_messages(ref_image_paths))
    config = _parse_prompt_config(raw, round_num)
    logger.info("Refined trigger (round %d): %s", round_num, config.trigger_phrase[:80])
    return config


async def refine_prompt_with_thinking(
    llm: LLMProvider,
    ref_image_paths: list[Path],
    current_trigger: str,
    round_num: int,
    ip_info: str,
    evaluations: list[RoundEvaluation],
    human_direction: str = "",
    thinking_budget: int = 5000,
) -> tuple[PromptConfig, str]:
    system_prompt = _build_system_prompt(
        current_trigger, round_num, ip_info, evaluations, human_direction,
    )
    response = await llm.invoke_with_thinking(
        system=system_prompt,
        messages=_build_messages(ref_image_paths),
        thinking_budget=thinking_budget,
    )
    config = _parse_prompt_config(response.text, round_num)
    logger.info("Refined trigger with thinking (round %d): %s", round_num, config.trigger_phrase[:80])
    return config, response.thinking
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run python -m pytest tests/agents/test_thinking_passthrough.py tests/agents/test_refine_prompt.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/agents/refine_prompt.py tests/agents/test_thinking_passthrough.py
git commit -m "feat(agents): refine_prompt_with_thinking returns reasoning"
```

---

### Task A6: Thread thinking through evaluate_result and select_model agents

**Files:**
- Modify: [src/styleclaw/agents/evaluate_result.py](src/styleclaw/agents/evaluate_result.py)
- Modify: [src/styleclaw/agents/select_model.py](src/styleclaw/agents/select_model.py)
- Modify: `tests/agents/test_thinking_passthrough.py` (append)

Apply the same pattern: split system-prompt/message helpers out, keep the existing public function, add a `_with_thinking` variant.

- [ ] **Step 1: Write the failing tests**

Append to `tests/agents/test_thinking_passthrough.py`:

```python
from styleclaw.agents.evaluate_result import (
    evaluate_round,
    evaluate_round_with_thinking,
)
from styleclaw.agents.select_model import (
    evaluate_models,
    evaluate_models_with_thinking,
)
from styleclaw.core.models import ModelEvaluation, RoundEvaluation


class TestEvaluateRoundWithThinking:
    async def test_returns_evaluation_and_thinking(self, ref_image, tmp_path):
        gen = tmp_path / "gen.png"
        from PIL import Image
        Image.new("RGB", (32, 32), "blue").save(gen)

        fake_llm = AsyncMock()
        fake_llm.invoke_with_thinking = AsyncMock(
            return_value=LLMResponse(
                text='{"evaluations": [], "recommendation": "keep refining"}',
                thinking="I compared color palettes...",
            )
        )
        evaluation, thinking = await evaluate_round_with_thinking(
            fake_llm, [ref_image], {"mj-v7": [gen]}, round_num=1,
            thinking_budget=3000,
        )
        assert isinstance(evaluation, RoundEvaluation)
        assert evaluation.round == 1
        assert thinking.startswith("I compared")


class TestEvaluateModelsWithThinking:
    async def test_returns_evaluation_and_thinking(self, ref_image, tmp_path):
        gen = tmp_path / "gen.png"
        from PIL import Image
        Image.new("RGB", (32, 32), "green").save(gen)

        fake_llm = AsyncMock()
        fake_llm.invoke_with_thinking = AsyncMock(
            return_value=LLMResponse(
                text='{"evaluations": [], "recommendation": "mj-v7"}',
                thinking="mj-v7 reproduced linework best.",
            )
        )
        evaluation, thinking = await evaluate_models_with_thinking(
            fake_llm, [ref_image], {"mj-v7/prompt-only": [gen]},
            thinking_budget=3000,
        )
        assert isinstance(evaluation, ModelEvaluation)
        assert evaluation.recommendation == "mj-v7"
        assert thinking.startswith("mj-v7 reproduced")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run python -m pytest tests/agents/test_thinking_passthrough.py -v`
Expected: FAIL — `_with_thinking` variants not yet defined.

- [ ] **Step 3: Modify evaluate_result.py**

Replace [src/styleclaw/agents/evaluate_result.py](src/styleclaw/agents/evaluate_result.py) with:

```python
from __future__ import annotations

import json
import logging
from pathlib import Path

from styleclaw.core.image_utils import build_image_block
from styleclaw.core.models import RoundEvaluation
from styleclaw.core.text_utils import clean_json, parse_llm_response
from styleclaw.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "providers" / "llm" / "prompts" / "evaluate.md"
)


def _build_system_prompt(round_num: int) -> str:
    return (
        PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("{round_num}", str(round_num))
    )


def _build_messages(
    ref_image_paths: list[Path],
    model_images: dict[str, list[Path]],
    round_num: int,
) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": "## Reference Images"}]
    content.extend(build_image_block(p) for p in ref_image_paths)
    for model_id, images in model_images.items():
        content.append({"type": "text", "text": f"## Generated by: {model_id} (Round {round_num})"})
        content.extend(build_image_block(p) for p in images)
    content.append({
        "type": "text",
        "text": "Score each model's output against the reference images across all 5 dimensions.",
    })
    return [{"role": "user", "content": content}]


def _parse_evaluation(raw_text: str, round_num: int) -> RoundEvaluation:
    cleaned = clean_json(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON for round evaluation: {exc}") from exc
    data["round"] = round_num
    return parse_llm_response(json.dumps(data), RoundEvaluation, "round evaluation")


async def evaluate_round(
    llm: LLMProvider,
    ref_image_paths: list[Path],
    model_images: dict[str, list[Path]],
    round_num: int,
) -> RoundEvaluation:
    raw = await llm.invoke(
        system=_build_system_prompt(round_num),
        messages=_build_messages(ref_image_paths, model_images, round_num),
        max_tokens=4096,
    )
    evaluation = _parse_evaluation(raw, round_num)
    logger.info("Round %d evaluation: recommendation=%s", round_num, evaluation.recommendation)
    return evaluation


async def evaluate_round_with_thinking(
    llm: LLMProvider,
    ref_image_paths: list[Path],
    model_images: dict[str, list[Path]],
    round_num: int,
    thinking_budget: int = 5000,
) -> tuple[RoundEvaluation, str]:
    response = await llm.invoke_with_thinking(
        system=_build_system_prompt(round_num),
        messages=_build_messages(ref_image_paths, model_images, round_num),
        max_tokens=4096,
        thinking_budget=thinking_budget,
    )
    evaluation = _parse_evaluation(response.text, round_num)
    logger.info(
        "Round %d evaluation (with thinking): recommendation=%s",
        round_num, evaluation.recommendation,
    )
    return evaluation, response.thinking
```

- [ ] **Step 4: Read and modify select_model.py**

First read it so you know the exact shape:

Run: `cat src/styleclaw/agents/select_model.py`

Then apply the same refactor pattern — extract `_build_system_prompt`, `_build_messages`, `_parse_evaluation` private helpers; keep `evaluate_models()`; add `evaluate_models_with_thinking()` that calls `invoke_with_thinking()` and returns `(ModelEvaluation, str)`.

Concretely, the new public function signature:

```python
async def evaluate_models_with_thinking(
    llm: LLMProvider,
    ref_image_paths: list[Path],
    model_images: dict[str, list[Path]],
    thinking_budget: int = 5000,
) -> tuple[ModelEvaluation, str]:
    response = await llm.invoke_with_thinking(
        system=_build_system_prompt(),
        messages=_build_messages(ref_image_paths, model_images),
        max_tokens=4096,
        thinking_budget=thinking_budget,
    )
    evaluation = _parse_evaluation(response.text)
    return evaluation, response.thinking
```

- [ ] **Step 5: Run tests to verify all pass**

Run: `uv run python -m pytest tests/agents/ -v`
Expected: PASS — all existing agent tests still pass, new thinking tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/styleclaw/agents/evaluate_result.py src/styleclaw/agents/select_model.py tests/agents/test_thinking_passthrough.py
git commit -m "feat(agents): evaluate_round/evaluate_models _with_thinking variants"
```

---

### Task A7: ExecutionContext gains thinking flags

**Files:**
- Modify: [src/styleclaw/orchestrator/actions.py](src/styleclaw/orchestrator/actions.py) — extend `ExecutionContext` only

- [ ] **Step 1: Write the failing test**

Append to `tests/orchestrator/test_actions.py` (existing file):

```python
class TestExecutionContextThinking:
    def test_thinking_defaults(self):
        from styleclaw.orchestrator.actions import ExecutionContext
        ctx = ExecutionContext(project="p")
        assert ctx.show_thinking is False
        assert ctx.thinking_budget == 5000

    def test_thinking_can_be_set(self):
        from styleclaw.orchestrator.actions import ExecutionContext
        ctx = ExecutionContext(project="p", show_thinking=True, thinking_budget=8000)
        assert ctx.show_thinking is True
        assert ctx.thinking_budget == 8000
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run python -m pytest tests/orchestrator/test_actions.py::TestExecutionContextThinking -v`
Expected: FAIL — unknown field `show_thinking`.

- [ ] **Step 3: Modify ExecutionContext**

In [src/styleclaw/orchestrator/actions.py](src/styleclaw/orchestrator/actions.py) near line 25, replace the `ExecutionContext` definition:

```python
@dataclass(frozen=True)
class ExecutionContext:
    project: str
    client: RunningHubClient | None = None
    llm: LLMProvider | None = None
    poll_interval: float = ORCHESTRATOR_POLL_INTERVAL
    show_thinking: bool = False
    thinking_budget: int = 5000
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/orchestrator/test_actions.py -v`
Expected: PASS — all tests still pass (dataclass keyword args are backward compatible).

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/orchestrator/actions.py tests/orchestrator/test_actions.py
git commit -m "feat(orchestrator): ExecutionContext carries show_thinking + thinking_budget"
```

---

### Task A8: Wire thinking into orchestrator actions

**Files:**
- Modify: [src/styleclaw/orchestrator/actions.py](src/styleclaw/orchestrator/actions.py) — `do_analyze`, `do_evaluate`, `do_refine` branches

When `ctx.show_thinking` is True, call the `_with_thinking` agent variant, save the thinking beside the JSON via `project_store.save_thinking()`, and include a short excerpt in the `StepResult.message`.

- [ ] **Step 1: Write the failing test**

Create `tests/orchestrator/test_actions_thinking.py`:

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from styleclaw.core.models import (
    Phase,
    ProjectConfig,
    ProjectState,
    PromptConfig,
    RoundEvaluation,
    StyleAnalysis,
)
from styleclaw.orchestrator.actions import ExecutionContext, do_analyze, do_refine
from styleclaw.providers.llm.base import LLMResponse
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


@pytest.fixture
def project_with_ref(tmp_path):
    name = "proj"
    root = tmp_path / "projects" / name
    root.mkdir(parents=True)
    (root / "refs").mkdir()
    ref = root / "refs" / "ref.png"
    Image.new("RGB", (32, 32), "red").save(ref)
    config = ProjectConfig(name=name, ip_info="test", ref_images=["refs/ref.png"])
    state = ProjectState(phase=Phase.INIT)
    (root / "model-select" / "results").mkdir(parents=True)
    (root / "style-refine").mkdir()
    (root / "batch-t2i").mkdir()
    (root / "batch-i2i").mkdir()
    project_store.save_config(name, config)
    project_store.save_state(name, state)
    return name


class TestDoAnalyzeThinking:
    async def test_saves_thinking_file_when_enabled(self, project_with_ref):
        fake_llm = AsyncMock()
        fake_llm.invoke_with_thinking = AsyncMock(
            return_value=LLMResponse(
                text='{"trigger_phrase": "bold ink"}',
                thinking="Reasoning text here.",
            )
        )
        ctx = ExecutionContext(
            project=project_with_ref, llm=fake_llm,
            show_thinking=True, thinking_budget=3000,
        )
        result = await do_analyze(ctx, {})
        assert result.ok

        thinking_md = (
            project_store.project_dir(project_with_ref)
            / "model-select" / "initial-analysis.thinking.md"
        )
        assert thinking_md.exists()
        assert "Reasoning text here." in thinking_md.read_text(encoding="utf-8")

    async def test_no_thinking_file_when_disabled(self, project_with_ref):
        fake_llm = AsyncMock()
        fake_llm.invoke = AsyncMock(return_value='{"trigger_phrase": "bold"}')
        ctx = ExecutionContext(project=project_with_ref, llm=fake_llm, show_thinking=False)
        result = await do_analyze(ctx, {})
        assert result.ok

        thinking_md = (
            project_store.project_dir(project_with_ref)
            / "model-select" / "initial-analysis.thinking.md"
        )
        assert not thinking_md.exists()
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run python -m pytest tests/orchestrator/test_actions_thinking.py -v`
Expected: FAIL — `do_analyze` doesn't look at `ctx.show_thinking` yet; no thinking file is written.

- [ ] **Step 3: Modify do_analyze**

In [src/styleclaw/orchestrator/actions.py](src/styleclaw/orchestrator/actions.py), replace the `do_analyze` function (currently ~lines 41–56) with:

```python
async def do_analyze(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.agents.analyze_style import analyze_style, analyze_style_with_thinking
    from styleclaw.core.state_machine import advance

    config = project_store.load_config(ctx.project)
    root = project_store.project_dir(ctx.project)
    ref_paths = [root / r for r in config.ref_images]

    thinking = ""
    if ctx.show_thinking:
        analysis, thinking = await analyze_style_with_thinking(
            ctx.llm, ref_paths, config.ip_info, thinking_budget=ctx.thinking_budget,
        )
    else:
        analysis = await analyze_style(ctx.llm, ref_paths, config.ip_info)
    project_store.save_analysis(ctx.project, analysis)

    if thinking:
        project_store.save_thinking(
            root / "model-select" / "initial-analysis.json", thinking,
        )

    state = project_store.load_state(ctx.project)
    new_state = advance(state, Phase.MODEL_SELECT)
    project_store.save_state(ctx.project, new_state)

    msg = f"Trigger: {analysis.trigger_phrase}"
    if thinking:
        msg += f" | thinking saved ({len(thinking)} chars)"
    return StepResult(ok=True, message=msg)
```

- [ ] **Step 4: Modify do_refine**

Replace `do_refine` (currently ~lines 214–252) with:

```python
async def do_refine(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.agents.refine_prompt import refine_prompt, refine_prompt_with_thinking
    from styleclaw.core.models import RoundEvaluation

    state = project_store.load_state(ctx.project)
    config = project_store.load_config(ctx.project)
    root = project_store.project_dir(ctx.project)
    ref_paths = [root / r for r in config.ref_images]

    round_num = state.current_round + 1
    if round_num > MAX_AUTO_ROUNDS:
        return StepResult(ok=False, message=f"Max rounds ({MAX_AUTO_ROUNDS}) reached")

    evaluations: list[RoundEvaluation] = []
    for r in range(1, round_num):
        try:
            ev = project_store.load_round_evaluation(ctx.project, r)
            evaluations.append(ev)
        except FileNotFoundError:
            logger.warning("Evaluation for round %d not found, skipping history entry.", r)

    if round_num == 1:
        analysis = project_store.load_analysis(ctx.project)
        current_trigger = analysis.trigger_phrase
    else:
        prev_prompt = project_store.load_prompt_config(ctx.project, round_num - 1)
        current_trigger = prev_prompt.trigger_phrase

    direction = args.get("direction", "")
    thinking = ""
    if ctx.show_thinking:
        prompt_config, thinking = await refine_prompt_with_thinking(
            ctx.llm, ref_paths, current_trigger, round_num,
            config.ip_info, evaluations, direction,
            thinking_budget=ctx.thinking_budget,
        )
    else:
        prompt_config = await refine_prompt(
            ctx.llm, ref_paths, current_trigger, round_num,
            config.ip_info, evaluations, direction,
        )
    project_store.save_prompt_config(ctx.project, round_num, prompt_config)

    if thinking:
        round_dir = project_store.round_dir(ctx.project, round_num)
        project_store.save_thinking(round_dir / "prompt.json", thinking)

    new_state = state.with_round(round_num)
    project_store.save_state(ctx.project, new_state)

    msg = f"Round {round_num}: {prompt_config.trigger_phrase}"
    if thinking:
        msg += f" | thinking saved ({len(thinking)} chars)"
    return StepResult(ok=True, message=msg)
```

- [ ] **Step 5: Modify do_evaluate**

In [src/styleclaw/orchestrator/actions.py](src/styleclaw/orchestrator/actions.py), replace both phase branches of `do_evaluate` (currently ~lines 120–184). Replace the full function with:

```python
async def do_evaluate(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    state = project_store.load_state(ctx.project)
    config = project_store.load_config(ctx.project)
    root = project_store.project_dir(ctx.project)
    ref_paths = [root / r for r in config.ref_images]

    if state.phase == Phase.MODEL_SELECT:
        from styleclaw.agents.select_model import (
            evaluate_models,
            evaluate_models_with_thinking,
        )
        from styleclaw.scripts.report import generate_model_select_report

        model_images: dict[str, list[Path]] = {}
        records = project_store.load_all_task_records(ctx.project)
        for key in records:
            if "/" in key:
                model_id, variant = key.split("/", 1)
                results_dir = project_store.model_results_dir(ctx.project, model_id, variant=variant)
            else:
                results_dir = project_store.model_results_dir(ctx.project, key)
            images = sorted(results_dir.glob("output-*.png"))
            if images:
                model_images[key] = images

        if not model_images:
            return StepResult(ok=False, message="No generated images found")

        thinking = ""
        if ctx.show_thinking:
            evaluation, thinking = await evaluate_models_with_thinking(
                ctx.llm, ref_paths, model_images, thinking_budget=ctx.thinking_budget,
            )
        else:
            evaluation = await evaluate_models(ctx.llm, ref_paths, model_images)
        project_store.save_evaluation(ctx.project, evaluation)
        if thinking:
            project_store.save_thinking(
                root / "model-select" / "evaluation.json", thinking,
            )
        generate_model_select_report(ctx.project)

        msg = f"Recommendation: {evaluation.recommendation}"
        if thinking:
            msg += f" | thinking saved ({len(thinking)} chars)"
        return StepResult(
            ok=True, message=msg,
            data={"recommendation": evaluation.recommendation},
        )

    if state.phase == Phase.STYLE_REFINE:
        from styleclaw.agents.evaluate_result import (
            evaluate_round,
            evaluate_round_with_thinking,
        )
        from styleclaw.scripts.report import generate_style_refine_report

        round_num = state.current_round
        model_images = {}
        records = project_store.load_all_round_task_records(ctx.project, round_num)
        for mid in records:
            results_dir = project_store.round_results_dir(ctx.project, round_num, mid)
            images = sorted(results_dir.glob("output-*.png"))
            if images:
                model_images[mid] = images

        if not model_images:
            return StepResult(ok=False, message="No generated images for this round")

        thinking = ""
        if ctx.show_thinking:
            evaluation, thinking = await evaluate_round_with_thinking(
                ctx.llm, ref_paths, model_images, round_num,
                thinking_budget=ctx.thinking_budget,
            )
        else:
            evaluation = await evaluate_round(ctx.llm, ref_paths, model_images, round_num)
        project_store.save_round_evaluation(ctx.project, round_num, evaluation)
        if thinking:
            round_d = project_store.round_dir(ctx.project, round_num)
            project_store.save_thinking(round_d / "evaluation.json", thinking)
        generate_style_refine_report(ctx.project, round_num)

        passed = evaluation.should_approve()
        scores_msg = ", ".join(
            f"{e.model}={e.total:.1f}" for e in evaluation.evaluations
        )
        msg = f"Scores: [{scores_msg}] {'PASS' if passed else 'needs refinement'}"
        if thinking:
            msg += f" | thinking saved ({len(thinking)} chars)"
        return StepResult(ok=True, message=msg, data={"passed": passed})

    return StepResult(ok=False, message=f"Cannot evaluate in {state.phase}")
```

- [ ] **Step 6: Run orchestrator tests**

Run: `uv run python -m pytest tests/orchestrator/ -v`
Expected: PASS — all existing orchestrator tests still pass plus the two new thinking tests.

- [ ] **Step 7: Commit**

```bash
git add src/styleclaw/orchestrator/actions.py tests/orchestrator/test_actions_thinking.py
git commit -m "feat(orchestrator): save LLM thinking sidecar when show_thinking enabled"
```

---

### Task A9: CLI `--show-thinking` flag and Typer wiring

**Files:**
- Modify: [src/styleclaw/cli.py](src/styleclaw/cli.py) — `_run_action`, `_build_context`, add flag to `analyze`, `refine`, `evaluate`, `run`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
class TestShowThinkingFlag:
    def test_analyze_forwards_show_thinking(self, setup_project, monkeypatch):
        from unittest.mock import MagicMock, patch
        from styleclaw.cli import app

        captured_ctx = {}

        async def fake_action(ctx, args):
            captured_ctx["show_thinking"] = ctx.show_thinking
            captured_ctx["thinking_budget"] = ctx.thinking_budget
            from styleclaw.orchestrator.actions import StepResult
            # Also advance the state so the post-condition check passes
            from styleclaw.core.models import Phase
            from styleclaw.core.state_machine import advance
            state = project_store.load_state(ctx.project)
            project_store.save_state(ctx.project, advance(state, Phase.MODEL_SELECT))
            return StepResult(ok=True, message="ok")

        with patch.dict(
            "styleclaw.orchestrator.actions.ACTION_REGISTRY",
            {"analyze": MagicMock(
                fn=fake_action, needs_client=False, needs_llm=True,
                requires_confirmation=False,
            )},
        ):
            # Stub BedrockProvider so _build_context doesn't need real creds
            with patch("styleclaw.providers.llm.bedrock.BedrockProvider") as FakeLLM:
                FakeLLM.return_value = MagicMock()
                FakeLLM.return_value.close = AsyncMock()
                result = CliRunner().invoke(
                    app, ["analyze", "test-project", "--show-thinking", "--thinking-budget", "4000"],
                )
        assert result.exit_code == 0
        assert captured_ctx["show_thinking"] is True
        assert captured_ctx["thinking_budget"] == 4000
```

(Note: adjust imports — the file already has `CliRunner`, `AsyncMock`, `project_store` fixtures; check the top of `tests/test_cli.py` before adding. If `AsyncMock` is missing, add `from unittest.mock import AsyncMock`.)

- [ ] **Step 2: Run test to verify failure**

Run: `uv run python -m pytest tests/test_cli.py::TestShowThinkingFlag -v`
Expected: FAIL — unknown option `--show-thinking`.

- [ ] **Step 3: Modify _build_context and _run_action**

In [src/styleclaw/cli.py](src/styleclaw/cli.py), change the signature of `_build_context` (line 33–54) to:

```python
@asynccontextmanager
async def _build_context(
    project: str,
    needs_client: bool = False,
    needs_llm: bool = False,
    show_thinking: bool = False,
    thinking_budget: int = 5000,
) -> AsyncIterator[ExecutionContext]:
    from styleclaw.providers.llm.bedrock import BedrockProvider
    from styleclaw.providers.runninghub.client import RunningHubClient

    client = None
    llm = None
    try:
        if needs_client:
            client = RunningHubClient(api_key=_get_api_key())
        if needs_llm:
            llm = BedrockProvider()
        yield ExecutionContext(
            project=project, client=client, llm=llm,
            show_thinking=show_thinking, thinking_budget=thinking_budget,
        )
    finally:
        if client:
            await client.close()
        if llm:
            await llm.close()
```

And update `_run_action` (line 57–88) to accept and forward the flags:

```python
def _run_action(
    project: str,
    action_name: str,
    args: dict[str, Any] | None = None,
    show_thinking: bool = False,
    thinking_budget: int = 5000,
) -> StepResult:
    import httpx

    from styleclaw.orchestrator.actions import ACTION_REGISTRY

    action_def = ACTION_REGISTRY.get(action_name)
    if action_def is None:
        raise ValueError(f"Unknown action: {action_name}")

    async def _exec() -> StepResult:
        async with _build_context(
            project,
            needs_client=action_def.needs_client,
            needs_llm=action_def.needs_llm,
            show_thinking=show_thinking,
            thinking_budget=thinking_budget,
        ) as ctx:
            return await action_def.fn(ctx, args or {})

    try:
        return asyncio.run(_exec())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except httpx.HTTPStatusError as exc:
        typer.echo(f"API error ({exc.response.status_code}): {exc}", err=True)
        raise typer.Exit(1) from exc
    except httpx.TransportError as exc:
        typer.echo(f"Network error: {exc}", err=True)
        raise typer.Exit(1) from exc
```

- [ ] **Step 4: Add --show-thinking to analyze, evaluate, refine, run**

Modify `analyze` command (~line 91–112). Replace with:

```python
@app.command()
def analyze(
    name: str = typer.Argument(..., help="Project name"),
    show_thinking: bool = typer.Option(
        False, "--show-thinking", help="Capture and save LLM reasoning alongside output",
    ),
    thinking_budget: int = typer.Option(
        5000, "--thinking-budget", help="Thinking token budget (when --show-thinking)",
    ),
) -> None:
    """Analyze reference images and generate initial trigger phrase."""
    state = project_store.load_state(name)
    if state.phase != Phase.INIT:
        typer.echo(f"Error: Project must be in INIT phase (current: {state.phase})", err=True)
        raise typer.Exit(1)

    result = _run_action(
        name, "analyze",
        show_thinking=show_thinking, thinking_budget=thinking_budget,
    )
    typer.echo(f"Analysis complete. {result.message}")
    if show_thinking:
        md = (
            project_store.project_dir(name)
            / "model-select" / "initial-analysis.thinking.md"
        )
        if md.exists():
            typer.echo("\n--- LLM thinking ---")
            typer.echo(md.read_text(encoding="utf-8"))
            typer.echo("--- end thinking ---\n")
    state = project_store.load_state(name)
    typer.echo(f"Phase advanced to: {state.phase}")
```

Apply the same pattern to `evaluate` (line 199–213) and `refine` (line 246–268): add the two options and forward them. After `evaluate`, the thinking sidecar path is `model-select/evaluation.thinking.md` (MODEL_SELECT) or `style-refine/round-NNN/evaluation.thinking.md` (STYLE_REFINE) — echo it when present. After `refine`, the path is `style-refine/round-NNN/prompt.thinking.md`.

For `run` (line 544), add the same options. It already uses `_build_context` internally (line 598) — update that call to pass `show_thinking=show_thinking, thinking_budget=thinking_budget`.

- [ ] **Step 5: Run tests**

Run: `uv run python -m pytest tests/test_cli.py -v`
Expected: PASS — new CLI test passes, existing tests still pass.

- [ ] **Step 6: Run the full suite**

Run: `uv run python -m pytest tests/ -v`
Expected: PASS — full green, coverage ≥ 80%.

- [ ] **Step 7: Commit**

```bash
git add src/styleclaw/cli.py tests/test_cli.py
git commit -m "feat(cli): --show-thinking flag captures and displays LLM reasoning"
```

---

### Part A Self-Review Checklist

Before moving to Part B, verify:

- [ ] `uv run python -m pytest tests/ -v` is fully green
- [ ] `uv run python -m pytest tests/ --cov=src` shows coverage ≥ 80%
- [ ] `uv run styleclaw analyze <test-project> --show-thinking` produces `initial-analysis.thinking.md` and echoes it
- [ ] `uv run styleclaw analyze <test-project>` (without flag) produces NO `.thinking.md` file (backward compatible)
- [ ] `invoke()` still works at temperature=0.3; `invoke_with_thinking()` forces temperature=1.0
- [ ] No orphan references to removed functions

---
