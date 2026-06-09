# StyleClaw 本地 Web UI — M1 后端实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付一个无头（headless）FastAPI 后端，把现有 orchestrator/`ACTION_REGISTRY` 暴露为 REST + WebSocket，用 `curl`/`websocat` 即可走完整条流水线、看实时进度（含 LLM 逐字流式）、断线重连——为 M2 的 React 前端铺好契约。

**Architecture:** FastAPI 应用包住现有 `actions`/`planner`/`executor`。所有「运行」（单动作或多步计划）统一经 `RunManager` 在后台 asyncio task 里跑，钩子 `on_step_start`/`on_step_done` 转成事件，缓冲 + 广播给 WebSocket 订阅者，支持重放重连。三个需确认动作（`init`/`select-model`/`add-refs`）走独立 REST 端点（表单/multipart 直接给齐参数），不进自动循环——绕开同步 `on_confirm` 在 async 里阻塞的问题。LLM 逐字流式通过 `core/stream_sink.py` 的 `ContextVar` 出口实现，CLI 行为不变。

**Tech Stack:** Python 3.11 / FastAPI / uvicorn / Pydantic v2 / pytest（starlette `TestClient`）。前端不在本计划内（M2）。

**Spec:** `docs/superpowers/specs/2026-06-09-styleclaw-web-ui-design.md`（M1 部分）

### 与 spec 的两处务实偏差（已确认理由）

1. **图库**：新建只读 `web/gallery.py`，复用底层 `list_output_images()` / `project_store.*_dir()`，**不重构** `report.py` 的 4 个渲染函数。降低风险，HTML 报告路径保持不动；完整 DRY 抽取推迟。
2. **Context 构建**：`web/context.py` 独立镜像 `cli._build_context`（约 20 行），避免 web↔cli 的导入环。完整共享化推迟。

---

## 文件结构

**新建：**
- `src/styleclaw/web/__init__.py` — 包标记
- `src/styleclaw/web/app.py` — FastAPI 工厂 `create_app()`，装配所有路由
- `src/styleclaw/web/context.py` — `build_context()` async contextmanager（构建/释放 `ExecutionContext`）
- `src/styleclaw/web/events.py` — Pydantic 事件 schema（前后端契约）
- `src/styleclaw/web/run_manager.py` — `RunManager`：运行生命周期 + 事件总线 + 重放 + 单运行锁
- `src/styleclaw/web/gallery.py` — `build_gallery()`：当前阶段图库 JSON（图片转 `/media/...` URL）
- `src/styleclaw/web/routes_projects.py` — 项目列表/详情/图库/媒体 端点
- `src/styleclaw/web/routes_runs.py` — plan 预览 / run 启动 / run 状态 / WS 事件
- `src/styleclaw/web/routes_confirm.py` — `init`/`select-model`/`add-refs` 确认端点
- `src/styleclaw/web/launch.py` — uvicorn 启动 + 自动开浏览器
- `src/styleclaw/core/stream_sink.py` — LLM 增量 `ContextVar` 出口
- `tests/web/__init__.py`
- `tests/web/conftest.py` — 共享夹具（tmp DATA_ROOT、TestClient、stub LLM/router）
- `tests/web/test_*.py` — 每个任务对应一个测试文件

**修改：**
- `pyproject.toml` — 加 `fastapi` / `uvicorn[standard]` / `python-multipart` 依赖；加 `styleclaw-web` 入口（可选）
- `src/styleclaw/providers/llm/openai_compat.py:116-123` — 流式循环改为优先走 delta sink
- `src/styleclaw/cli.py` — 新增 `web` 命令

---

## Task 1: 依赖 + 包骨架 + 应用工厂 + 健康检查

**Files:**
- Modify: `pyproject.toml`
- Create: `src/styleclaw/web/__init__.py`
- Create: `src/styleclaw/web/app.py`
- Create: `tests/web/__init__.py`
- Create: `tests/web/conftest.py`
- Test: `tests/web/test_app.py`

- [ ] **Step 1: 加依赖**

修改 `pyproject.toml` 的 `dependencies` 列表，在 `"filelock>=3.29.0",` 后追加：

```toml
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "python-multipart>=0.0.9",
```

- [ ] **Step 2: 同步依赖**

Run: `uv sync`
Expected: 安装 fastapi / uvicorn / python-multipart，无错误。

- [ ] **Step 3: 写失败测试**

Create `tests/web/__init__.py`（空文件）。

Create `tests/web/conftest.py`:

```python
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from styleclaw.storage import project_store


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Isolate DATA_ROOT per test (mirrors tests/test_cli.py::use_tmp_data_root)."""
    root = tmp_path / "projects"
    monkeypatch.setattr(project_store, "DATA_ROOT", root)
    monkeypatch.setenv("RUNNINGHUB_API_KEY", "test-key")
    monkeypatch.setenv("STYLECLAW_SKIP_ENV_CHECK", "1")
    return root


@pytest.fixture
def client(data_root):
    from styleclaw.web.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
```

Create `tests/web/test_app.py`:

```python
def test_health_ok(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
```

- [ ] **Step 4: 跑测试确认失败**

Run: `uv run python -m pytest tests/web/test_app.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'styleclaw.web'`

- [ ] **Step 5: 写最小实现**

Create `src/styleclaw/web/__init__.py`（空文件）。

Create `src/styleclaw/web/app.py`:

```python
from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Build the StyleClaw local web app.

    Single-user, bound to 127.0.0.1 by the launcher. No auth by design.
    """
    app = FastAPI(title="StyleClaw", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
```

- [ ] **Step 6: 跑测试确认通过**

Run: `uv run python -m pytest tests/web/test_app.py -v`
Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add pyproject.toml uv.lock src/styleclaw/web/__init__.py src/styleclaw/web/app.py tests/web/
git commit -m "feat(web): app factory + health endpoint + test deps"
```

---

## Task 2: web/context.py — ExecutionContext 构建器

**Files:**
- Create: `src/styleclaw/web/context.py`
- Test: `tests/web/test_context.py`

- [ ] **Step 1: 写失败测试**

Create `tests/web/test_context.py`:

```python
import pytest

from styleclaw.web.context import build_context


@pytest.mark.asyncio
async def test_context_without_client_or_llm(data_root):
    async with build_context("proj", needs_client=False, needs_llm=False) as ctx:
        assert ctx.project == "proj"
        assert ctx.client is None
        assert ctx.llm_router is None


@pytest.mark.asyncio
async def test_context_reuses_passed_router(data_root):
    sentinel = object()
    async with build_context("proj", needs_llm=True, router=sentinel) as ctx:
        assert ctx.llm_router is sentinel
    # passed-in router is NOT closed by build_context (caller owns it)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/web/test_context.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'styleclaw.web.context'`

- [ ] **Step 3: 写实现**

Create `src/styleclaw/web/context.py`:

```python
from __future__ import annotations

import asyncio
import inspect
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from styleclaw.orchestrator.actions import ExecutionContext

logger = logging.getLogger(__name__)


async def _close_resource(resource: Any, label: str) -> None:
    close = getattr(resource, "close", None)
    if close is None:
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await asyncio.wait_for(result, timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("Timed out closing %s after 5s.", label)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error closing %s: %s", label, exc)


@asynccontextmanager
async def build_context(
    project: str,
    *,
    needs_client: bool = False,
    needs_llm: bool = False,
    router: Any = None,
) -> AsyncIterator[ExecutionContext]:
    """Build an ExecutionContext for a web request/run.

    Mirrors ``cli._build_context`` but standalone (no Typer dependency) to
    avoid a web<->cli import cycle. A ``router`` passed in by the caller is
    reused and NOT closed here (caller owns its lifecycle).
    """
    from styleclaw.core.llm_routing import RoleRouter
    from styleclaw.providers.runninghub.client import RunningHubClient

    client = None
    owns_router = False
    try:
        if needs_client:
            key = os.getenv("RUNNINGHUB_API_KEY")
            if not key:
                raise RuntimeError("RUNNINGHUB_API_KEY not set")
            client = RunningHubClient(api_key=key)
        if needs_llm and router is None:
            router = RoleRouter.from_env()
            owns_router = True
        yield ExecutionContext(project=project, client=client, llm_router=router)
    finally:
        if client is not None:
            await _close_resource(client, "client")
        if router is not None and owns_router:
            await _close_resource(router, "llm_router")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run python -m pytest tests/web/test_context.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/styleclaw/web/context.py tests/web/test_context.py
git commit -m "feat(web): standalone ExecutionContext builder"
```

---

## Task 3: events.py — 事件 schema

**Files:**
- Create: `src/styleclaw/web/events.py`
- Test: `tests/web/test_events.py`

- [ ] **Step 1: 写失败测试**

Create `tests/web/test_events.py`:

```python
from styleclaw.web.events import (
    DoneEvent,
    ErrorEvent,
    LlmDeltaEvent,
    NeedsHumanEvent,
    PhasePausedEvent,
    RunStartedEvent,
    StepDoneEvent,
    StepStartEvent,
)


def test_event_has_type_tag():
    ev = StepStartEvent(index=0, name="analyze", description="分析风格")
    d = ev.model_dump()
    assert d["type"] == "step_start"
    assert d["index"] == 0
    assert d["name"] == "analyze"


def test_all_events_carry_distinct_type():
    events = [
        RunStartedEvent(run_id="r1", project="p", kind="plan", steps=["analyze"]),
        StepStartEvent(index=0, name="analyze", description="d"),
        LlmDeltaEvent(step_index=0, role="vision_analyst", text="abc"),
        StepDoneEvent(index=0, name="analyze", status="ok", summary="done"),
        NeedsHumanEvent(round=2, weakest_dim="color", score=4.2, suggestion="提高对比"),
        PhasePausedEvent(phase="MODEL_SELECT", next_phase="STYLE_REFINE"),
        DoneEvent(run_id="r1"),
        ErrorEvent(message="boom", detail="trace"),
    ]
    types = [e.model_dump()["type"] for e in events]
    assert types == [
        "run_started", "step_start", "llm_delta", "step_done",
        "needs_human", "phase_paused", "done", "error",
    ]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/web/test_events.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'styleclaw.web.events'`

- [ ] **Step 3: 写实现**

Create `src/styleclaw/web/events.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class _Event(BaseModel):
    model_config = {"frozen": True}


class RunStartedEvent(_Event):
    type: Literal["run_started"] = "run_started"
    run_id: str
    project: str
    kind: str  # "plan" | "phase" | "action"
    steps: list[str]


class StepStartEvent(_Event):
    type: Literal["step_start"] = "step_start"
    index: int
    name: str
    description: str


class LlmDeltaEvent(_Event):
    type: Literal["llm_delta"] = "llm_delta"
    step_index: int
    role: str
    text: str


class StepDoneEvent(_Event):
    type: Literal["step_done"] = "step_done"
    index: int
    name: str
    status: str  # "ok" | "fail"
    summary: str


class NeedsHumanEvent(_Event):
    type: Literal["needs_human"] = "needs_human"
    round: int
    weakest_dim: str
    score: float
    suggestion: str


class PhasePausedEvent(_Event):
    type: Literal["phase_paused"] = "phase_paused"
    phase: str
    next_phase: str


class DoneEvent(_Event):
    type: Literal["done"] = "done"
    run_id: str


class ErrorEvent(_Event):
    type: Literal["error"] = "error"
    message: str
    detail: str = ""
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run python -m pytest tests/web/test_events.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/styleclaw/web/events.py tests/web/test_events.py
git commit -m "feat(web): WebSocket event schema"
```

---

## Task 4: core/stream_sink.py — LLM 增量出口

**Files:**
- Create: `src/styleclaw/core/stream_sink.py`
- Test: `tests/core/test_stream_sink.py`

- [ ] **Step 1: 写失败测试**

Create `tests/core/test_stream_sink.py`:

```python
from styleclaw.core.stream_sink import emit_delta, reset_delta_sink, set_delta_sink


def test_emit_returns_false_when_no_sink():
    assert emit_delta("x") is False


def test_emit_routes_to_sink():
    captured = []
    token = set_delta_sink(captured.append)
    try:
        assert emit_delta("hello") is True
        assert emit_delta(" world") is True
    finally:
        reset_delta_sink(token)
    assert captured == ["hello", " world"]
    # after reset, no sink again
    assert emit_delta("z") is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/core/test_stream_sink.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 写实现**

Create `src/styleclaw/core/stream_sink.py`:

```python
from __future__ import annotations

import contextvars
from typing import Callable

# A sink receives raw text deltas as they stream from the LLM provider.
DeltaSink = Callable[[str], None]

_current_sink: contextvars.ContextVar[DeltaSink | None] = contextvars.ContextVar(
    "styleclaw_delta_sink", default=None,
)


def set_delta_sink(sink: DeltaSink | None) -> contextvars.Token:
    """Install a delta sink for the current context. Returns a token for reset."""
    return _current_sink.set(sink)


def reset_delta_sink(token: contextvars.Token) -> None:
    _current_sink.reset(token)


def emit_delta(text: str) -> bool:
    """Send a streaming delta to the active sink if one is installed.

    Returns True if a sink consumed it, False otherwise (callers then fall
    back to their default behavior, e.g. printing to stdout).
    """
    sink = _current_sink.get()
    if sink is None:
        return False
    sink(text)
    return True
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run python -m pytest tests/core/test_stream_sink.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/styleclaw/core/stream_sink.py tests/core/test_stream_sink.py
git commit -m "feat(core): ContextVar delta sink for LLM streaming"
```

---

## Task 5: 把 delta sink 接入 openai_compat 流式循环

**Files:**
- Modify: `src/styleclaw/providers/llm/openai_compat.py:116-123`
- Test: `tests/providers/test_stream_sink_wiring.py`

- [ ] **Step 1: 写失败测试**

Create `tests/providers/__init__.py` 如果不存在（已存在则跳过）。

Create `tests/providers/test_stream_sink_wiring.py`:

```python
import httpx
import pytest
import respx

from styleclaw.core.stream_sink import reset_delta_sink, set_delta_sink
from styleclaw.providers.llm.openai_compat import OpenAICompatProvider


def _sse(*deltas: str) -> str:
    lines = []
    for d in deltas:
        import json
        payload = json.dumps({"choices": [{"delta": {"content": d}}]})
        lines.append(f"data: {payload}")
    lines.append("data: [DONE]")
    return "\n".join(lines) + "\n"


@pytest.mark.asyncio
@respx.mock
async def test_sink_receives_deltas():
    respx.post("https://fake.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse("Hel", "lo"),
        )
    )
    provider = OpenAICompatProvider(
        base_url="https://fake.test/v1", api_key="k", model_id="m",
    )
    captured: list[str] = []
    token = set_delta_sink(captured.append)
    try:
        text = await provider.invoke("sys", [{"role": "user", "content": "hi"}])
    finally:
        reset_delta_sink(token)
        await provider.close()
    assert text == "Hello"
    assert captured == ["Hel", "lo"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/providers/test_stream_sink_wiring.py -v`
Expected: FAIL — `captured == []`（当前 delta 只走 `print`，没接 sink）

- [ ] **Step 3: 改实现**

在 `src/styleclaw/providers/llm/openai_compat.py` 顶部 import 区（第 19 行 `from styleclaw.providers.llm._retry import llm_retry_loop` 下方）加：

```python
from styleclaw.core.stream_sink import emit_delta
```

把第 116-121 行的这段：

```python
                            if STREAM_DISPLAY:
                                if not stream_started:
                                    print("  ↓ ", end="", flush=True)
                                    stream_started = True
                                print(delta, end="", flush=True)
                            chunks.append(delta)
```

替换为：

```python
                            # Prefer the in-context delta sink (web UI). Only
                            # fall back to stdout printing when no sink is set,
                            # so CLI behavior is unchanged.
                            if not emit_delta(delta) and STREAM_DISPLAY:
                                if not stream_started:
                                    print("  ↓ ", end="", flush=True)
                                    stream_started = True
                                print(delta, end="", flush=True)
                            chunks.append(delta)
```

（第 122-123 行的 `if STREAM_DISPLAY and stream_started: print()` 保持不变。）

- [ ] **Step 4: 跑测试确认通过 + 回归**

Run: `uv run python -m pytest tests/providers/test_stream_sink_wiring.py tests/agents/ -v`
Expected: PASS（新测试通过；agents 测试不回归）

- [ ] **Step 5: 提交**

```bash
git add src/styleclaw/providers/llm/openai_compat.py tests/providers/test_stream_sink_wiring.py
git commit -m "feat(llm): route streaming deltas through delta sink"
```

---

## Task 6: run_manager.py — 运行生命周期 + 事件总线

**Files:**
- Create: `src/styleclaw/web/run_manager.py`
- Test: `tests/web/test_run_manager.py`

设计要点：
- `RunManager` 持有 `run_id -> _Run`，以及 `project -> 活跃 run_id`。
- `start(project, plan, kind)`：若该项目已有 running 运行 → 抛 `RunConflict`。否则建 `_Run`，`asyncio.create_task` 后台跑 `execute()`，立即返回 `run_id`。
- 钩子（同步）`on_step_start`/`on_step_done` → 调 `_emit`，把事件 append 到有界缓冲并 `put_nowait` 给每个订阅 `asyncio.Queue`。
- delta sink：非 panel 模式时安装，闭包捕获「当前 step_index」。
- `subscribe(run_id)` 返回 `(queue, replay_list)`；`unsubscribe` 移除。
- `get(run_id)` 返回 `{run_id, project, status, events}` 供重连。

- [ ] **Step 1: 写失败测试**

Create `tests/web/test_run_manager.py`:

```python
import asyncio

import pytest

from styleclaw.core.models import Action, ActionPlan, Phase, ProjectConfig, ProjectState
from styleclaw.storage import project_store
from styleclaw.web.run_manager import RunConflict, RunManager


@pytest.fixture
def refine_project(data_root):
    """A project parked in STYLE_REFINE so `approve` (no client/llm) runs cleanly."""
    config = ProjectConfig(name="p", ip_info="anime", ref_images=["refs/ref-001.png"])
    project_store.create_project(config)
    project_store.save_state("p", ProjectState(phase=Phase.STYLE_REFINE, current_round=1))
    return "p"


def _approve_plan() -> ActionPlan:
    return ActionPlan(
        summary="approve",
        steps=[Action(name="approve", description="approve", args={"target": "batch-t2i"})],
        loop=None,
        stop_summary="",
    )


@pytest.mark.asyncio
async def test_run_emits_step_and_done_events(refine_project):
    mgr = RunManager()
    run_id = await mgr.start(refine_project, _approve_plan(), kind="action")
    # wait for completion
    for _ in range(100):
        snap = mgr.get(run_id)
        if snap["status"] in ("done", "error"):
            break
        await asyncio.sleep(0.02)
    snap = mgr.get(run_id)
    assert snap["status"] == "done"
    types = [e["type"] for e in snap["events"]]
    assert "run_started" in types
    assert "step_start" in types
    assert "step_done" in types
    assert types[-1] == "done"
    # phase actually advanced
    assert project_store.load_state(refine_project).phase == Phase.BATCH_T2I


@pytest.mark.asyncio
async def test_second_run_while_active_conflicts(refine_project, monkeypatch):
    mgr = RunManager()

    # Make the action slow so the first run is still active when the second starts.
    import styleclaw.web.run_manager as rm

    real_execute = rm.execute

    async def slow_execute(plan, ctx, **kw):
        await asyncio.sleep(0.3)
        return await real_execute(plan, ctx, **kw)

    monkeypatch.setattr(rm, "execute", slow_execute)

    run_id = await mgr.start(refine_project, _approve_plan(), kind="action")
    with pytest.raises(RunConflict):
        await mgr.start(refine_project, _approve_plan(), kind="action")
    # let the first finish to avoid a dangling task
    for _ in range(100):
        if mgr.get(run_id)["status"] in ("done", "error"):
            break
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_subscribe_receives_live_events(refine_project):
    mgr = RunManager()
    run_id = await mgr.start(refine_project, _approve_plan(), kind="action")
    queue, replay = mgr.subscribe(run_id)
    seen = [e["type"] for e in replay]
    try:
        while "done" not in seen:
            ev = await asyncio.wait_for(queue.get(), timeout=2.0)
            seen.append(ev["type"])
    finally:
        mgr.unsubscribe(run_id, queue)
    assert "done" in seen
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/web/test_run_manager.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'styleclaw.web.run_manager'`

- [ ] **Step 3: 写实现**

Create `src/styleclaw/web/run_manager.py`:

```python
from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from typing import Any

import styleclaw.core.config as _cfg
from styleclaw.core.models import ActionPlan
from styleclaw.core.stream_sink import reset_delta_sink, set_delta_sink
from styleclaw.orchestrator.actions import ACTION_REGISTRY, StepResult
from styleclaw.orchestrator.executor import execute
from styleclaw.web.context import build_context
from styleclaw.web.events import (
    DoneEvent,
    ErrorEvent,
    LlmDeltaEvent,
    RunStartedEvent,
    StepDoneEvent,
    StepStartEvent,
)

logger = logging.getLogger(__name__)

_MAX_EVENTS = 2000


class RunConflict(RuntimeError):
    """Raised when a project already has an active (running) run."""


class _Run:
    def __init__(self, run_id: str, project: str) -> None:
        self.run_id = run_id
        self.project = project
        self.status = "running"  # running | done | error
        self.events: deque[dict] = deque(maxlen=_MAX_EVENTS)
        self.subscribers: set[asyncio.Queue] = set()
        self.current_step = 0
        self.task: asyncio.Task | None = None

    def emit(self, event_dict: dict) -> None:
        self.events.append(event_dict)
        for q in list(self.subscribers):
            try:
                q.put_nowait(event_dict)
            except asyncio.QueueFull:  # pragma: no cover - unbounded queues
                pass


def _plan_needs(plan: ActionPlan) -> tuple[bool, bool]:
    needs_client = False
    needs_llm = False
    for step in plan.steps:
        d = ACTION_REGISTRY.get(step.name)
        if d is None:
            continue
        needs_client = needs_client or d.needs_client
        needs_llm = needs_llm or d.needs_llm
    return needs_client, needs_llm


def _panel_active() -> bool:
    return bool(
        _cfg.PANEL_REFINE_ENABLED
        or _cfg.PANEL_MODEL_SELECT_ENABLED
        or _cfg.PANEL_ANALYZE_ENABLED
    )


class RunManager:
    def __init__(self) -> None:
        self._runs: dict[str, _Run] = {}
        self._active: dict[str, str] = {}  # project -> run_id

    def active_run_id(self, project: str) -> str | None:
        rid = self._active.get(project)
        if rid and self._runs.get(rid) and self._runs[rid].status == "running":
            return rid
        return None

    async def start(self, project: str, plan: ActionPlan, *, kind: str) -> str:
        if self.active_run_id(project) is not None:
            raise RunConflict(f"project '{project}' already has an active run")
        run_id = uuid.uuid4().hex
        run = _Run(run_id, project)
        self._runs[run_id] = run
        self._active[project] = run_id
        run.emit(
            RunStartedEvent(
                run_id=run_id, project=project, kind=kind,
                steps=[s.name for s in plan.steps],
            ).model_dump()
        )
        run.task = asyncio.create_task(self._execute(run, plan))
        return run_id

    async def _execute(self, run: _Run, plan: ActionPlan) -> None:
        needs_client, needs_llm = _plan_needs(plan)

        def on_step_start(index: int, name: str, description: str) -> None:
            run.current_step = index
            run.emit(StepStartEvent(index=index, name=name, description=description).model_dump())

        def on_step_done(index: int, name: str, result: StepResult) -> None:
            run.emit(
                StepDoneEvent(
                    index=index, name=name,
                    status="ok" if result.ok else "fail",
                    summary=result.message,
                ).model_dump()
            )

        sink_token = None
        try:
            async with build_context(
                run.project, needs_client=needs_client, needs_llm=needs_llm,
            ) as ctx:
                if not _panel_active():
                    def _sink(text: str) -> None:
                        run.emit(
                            LlmDeltaEvent(
                                step_index=run.current_step, role="", text=text,
                            ).model_dump()
                        )
                    sink_token = set_delta_sink(_sink)
                await execute(
                    plan, ctx,
                    on_step_start=on_step_start,
                    on_step_done=on_step_done,
                )
            run.status = "done"
            run.emit(DoneEvent(run_id=run.run_id).model_dump())
        except Exception as exc:  # noqa: BLE001
            logger.exception("run %s failed", run.run_id)
            run.status = "error"
            run.emit(ErrorEvent(message=str(exc), detail=type(exc).__name__).model_dump())
        finally:
            if sink_token is not None:
                reset_delta_sink(sink_token)
            if self._active.get(run.project) == run.run_id:
                self._active.pop(run.project, None)

    def get(self, run_id: str) -> dict[str, Any]:
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        return {
            "run_id": run.run_id,
            "project": run.project,
            "status": run.status,
            "events": list(run.events),
        }

    def subscribe(self, run_id: str) -> tuple[asyncio.Queue, list[dict]]:
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        q: asyncio.Queue = asyncio.Queue()
        replay = list(run.events)
        run.subscribers.add(q)
        return q, replay

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        run = self._runs.get(run_id)
        if run is not None:
            run.subscribers.discard(queue)
```

> **注意（给执行者）**：`set_delta_sink` 用 `ContextVar`，它在 `asyncio.create_task` 创建子任务时复制当前上下文。本实现里 sink 是在后台 task 内部（`_execute`）安装的，`execute()` 的 LLM 调用在同一 task 内 `await`，因此 sink 可见。panel/并发分支已通过 `_panel_active()` 守卫跳过，避免多模型 token 交错。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run python -m pytest tests/web/test_run_manager.py -v`
Expected: PASS（3 个测试）

- [ ] **Step 5: 提交**

```bash
git add src/styleclaw/web/run_manager.py tests/web/test_run_manager.py
git commit -m "feat(web): RunManager with event bus and single-run guard"
```

---

## Task 7: 项目列表 + 详情端点

**Files:**
- Create: `src/styleclaw/web/routes_projects.py`
- Modify: `src/styleclaw/web/app.py`
- Test: `tests/web/test_routes_projects.py`

- [ ] **Step 1: 写失败测试**

Create `tests/web/test_routes_projects.py`:

```python
from styleclaw.core.models import Phase, ProjectConfig, ProjectState
from styleclaw.storage import project_store


def _make_project(name: str, phase: Phase = Phase.INIT) -> None:
    project_store.create_project(
        ProjectConfig(name=name, ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state(name, ProjectState(phase=phase))


def test_list_projects_empty(client):
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    assert resp.json() == {"projects": []}


def test_list_projects(client, data_root):
    _make_project("alpha", Phase.MODEL_SELECT)
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    names = [p["name"] for p in resp.json()["projects"]]
    assert "alpha" in names
    alpha = next(p for p in resp.json()["projects"] if p["name"] == "alpha")
    assert alpha["phase"] == "MODEL_SELECT"


def test_project_detail(client, data_root):
    _make_project("beta", Phase.STYLE_REFINE)
    resp = client.get("/api/projects/beta")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"]["phase"] == "STYLE_REFINE"
    assert body["config"]["ip_info"] == "anime"
    assert isinstance(body["suggestions"], list)


def test_project_detail_not_found(client, data_root):
    resp = client.get("/api/projects/ghost")
    assert resp.status_code == 404
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/web/test_routes_projects.py -v`
Expected: FAIL — 404 on `/api/projects`（路由还没注册）

- [ ] **Step 3: 写实现**

Create `src/styleclaw/web/routes_projects.py`:

```python
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from styleclaw.orchestrator.suggestions import suggest_next_steps
from styleclaw.storage import project_store

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("")
async def list_projects() -> dict:
    out = []
    for name in project_store.list_projects():
        try:
            state = project_store.load_state(name)
        except FileNotFoundError:
            continue
        out.append({
            "name": name,
            "phase": state.phase.value,
            "current_round": state.current_round,
            "current_batch": state.current_batch,
            "last_updated": state.last_updated,
        })
    return {"projects": out}


@router.get("/{name}")
async def project_detail(name: str) -> dict:
    try:
        state = project_store.load_state(name)
        config = project_store.load_config(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"project '{name}' not found")
    return {
        "state": state.model_dump(),
        "config": config.model_dump(),
        "suggestions": suggest_next_steps(name),
    }
```

修改 `src/styleclaw/web/app.py`，在 `create_app()` 内 `return app` 之前加：

```python
    from styleclaw.web.routes_projects import router as projects_router
    app.include_router(projects_router)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run python -m pytest tests/web/test_routes_projects.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/styleclaw/web/routes_projects.py src/styleclaw/web/app.py tests/web/test_routes_projects.py
git commit -m "feat(web): project list and detail endpoints"
```

---

## Task 8: 图库 + 媒体端点

**Files:**
- Create: `src/styleclaw/web/gallery.py`
- Modify: `src/styleclaw/web/routes_projects.py`
- Test: `tests/web/test_gallery.py`

`build_gallery(name)` 按当前 phase 返回结构化 JSON。图片路径转 `/media/{name}/<rel>`（rel 相对 `project_dir(name)`）。

- [ ] **Step 1: 写失败测试**

Create `tests/web/test_gallery.py`:

```python
from styleclaw.core.models import Phase, ProjectConfig, ProjectState
from styleclaw.storage import project_store
from styleclaw.web.gallery import build_gallery


def _project_with_ref(name: str, phase: Phase) -> None:
    project_store.create_project(
        ProjectConfig(name=name, ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state(name, ProjectState(phase=phase))
    # write a dummy ref file so media URLs resolve
    ref = project_store.project_dir(name) / "refs" / "ref-001.png"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(b"\x89PNG\r\n\x1a\n")


def test_gallery_init_phase_lists_refs(data_root):
    _project_with_ref("g", Phase.INIT)
    gallery = build_gallery("g")
    assert gallery["phase"] == "INIT"
    assert gallery["ref_images"] == ["/media/g/refs/ref-001.png"]
    assert gallery["groups"] == []


def test_media_url_is_relative_to_project_dir(data_root):
    _project_with_ref("g", Phase.INIT)
    gallery = build_gallery("g")
    assert gallery["ref_images"][0].startswith("/media/g/")


def test_gallery_endpoint(client, data_root):
    _project_with_ref("g", Phase.INIT)
    resp = client.get("/api/projects/g/gallery")
    assert resp.status_code == 200
    assert resp.json()["phase"] == "INIT"


def test_media_endpoint_serves_file(client, data_root):
    _project_with_ref("g", Phase.INIT)
    resp = client.get("/media/g/refs/ref-001.png")
    assert resp.status_code == 200
    assert resp.content.startswith(b"\x89PNG")


def test_safe_media_path_rejects_traversal(data_root):
    # Test the guard directly: an httpx client normalizes `../` out of the URL
    # before it reaches the route, so only a direct call exercises this branch.
    from styleclaw.web.routes_projects import safe_media_path

    _project_with_ref("g", Phase.INIT)
    assert safe_media_path("g", "refs/ref-001.png") is not None
    assert safe_media_path("g", "../../../etc/passwd") is None


def test_gallery_model_select_attaches_scores(data_root):
    # Result-bearing MODEL_SELECT branch: a gender-suffixed task record must
    # still join to the base-variant evaluation score (regression guard).
    from styleclaw.core.models import (
        DimensionScores,
        ModelEvaluation,
        ModelScore,
        TaskRecord,
        TaskStatus,
    )

    project_store.create_project(
        ProjectConfig(name="m", ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state("m", ProjectState(phase=Phase.MODEL_SELECT, current_model_select_pass=1))

    rec = TaskRecord(task_id="t1", model_id="mj-v7", status=TaskStatus.SUCCESS)
    project_store.save_task_record("m", "mj-v7", rec, variant="prompt-sref-male", pass_num=1)
    results_dir = project_store.model_results_dir("m", "mj-v7", variant="prompt-sref-male", pass_num=1)
    (results_dir / "output-001.png").write_bytes(b"\x89PNG")

    evaluation = ModelEvaluation(
        evaluations=[
            ModelScore(
                model="mj-v7", variant="prompt-sref", total=8.0,
                scores=DimensionScores(visual_style=8.0),
            )
        ],
        recommendation="mj-v7",
    )
    project_store.save_evaluation("m", evaluation, pass_num=1)

    gallery = build_gallery("m")
    assert gallery["phase"] == "MODEL_SELECT"
    grp = next(g for g in gallery["groups"] if g["label"] == "mj-v7/prompt-sref-male")
    assert len(grp["images"]) == 1
    assert grp["scores"] is not None
    assert grp["scores"]["total"] == 8.0


def test_gallery_batch_t2i_lists_case_images(data_root):
    from styleclaw.core.models import BatchCase, BatchConfig

    project_store.create_project(
        ProjectConfig(name="b", ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state("b", ProjectState(phase=Phase.BATCH_T2I, current_batch=1))
    project_store.save_batch_config(
        "b", 1,
        BatchConfig(
            batch=1, trigger_phrase="trig",
            cases=[BatchCase(id="case-001", category="adult_male", description="d")],
        ),
    )
    case_dir = project_store.batch_t2i_case_dir("b", 1, "case-001")
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "output-001.png").write_bytes(b"\x89PNG")

    gallery = build_gallery("b")
    assert gallery["phase"] == "BATCH_T2I"
    assert gallery["groups"][0]["images"] == ["/media/b/batch-t2i/batch-001/results/case-001/output-001.png"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/web/test_gallery.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'styleclaw.web.gallery'`

- [ ] **Step 3: 写 gallery 实现**

Create `src/styleclaw/web/gallery.py`:

```python
from __future__ import annotations

from pathlib import Path

from styleclaw.core.models import Phase
from styleclaw.storage import project_store
from styleclaw.storage.image_store import list_output_images


def _media_url(name: str, path: Path) -> str:
    rel = path.relative_to(project_store.project_dir(name))
    return f"/media/{name}/{rel.as_posix()}"


def _ref_urls(name: str, config) -> list[str]:
    root = project_store.project_dir(name)
    urls = []
    for r in config.ref_images:
        p = root / r
        if p.exists():
            urls.append(_media_url(name, p))
    return urls


def build_gallery(name: str) -> dict:
    """Return a JSON-serializable gallery for the project's current phase.

    Shape:
        {phase, ref_images: [url], groups: [{label, images: [url], scores: {..}}]}
    """
    state = project_store.load_state(name)
    config = project_store.load_config(name)
    refs = _ref_urls(name, config)
    groups: list[dict] = []

    if state.phase == Phase.MODEL_SELECT:
        pass_num = state.current_model_select_pass or 1
        try:
            evaluation = project_store.load_evaluation(name, pass_num=pass_num)
        except FileNotFoundError:
            evaluation = None
        # Evaluation scores are keyed by (model, BASE variant) — e.g.
        # "mj-v7/prompt-sref". But task-record keys carry a gender suffix
        # ("mj-v7/prompt-sref-male", see do_evaluate / report.py:67). Strip the
        # suffix when joining or every score comes back None.
        score_by_key: dict[str, dict] = {}
        if evaluation is not None:
            for e in evaluation.evaluations:
                k = f"{e.model}/{e.variant}" if e.variant else e.model
                score_by_key[k] = {"total": e.total, **e.scores.model_dump()}
        records = project_store.load_all_task_records(name, pass_num=pass_num)
        for rec_key in sorted(records):
            if "/" in rec_key:
                model_id, variant = rec_key.split("/", 1)
                results_dir = project_store.model_results_dir(
                    name, model_id, variant=variant, pass_num=pass_num,
                )
                base_variant = variant
                for suffix in ("-male", "-female"):
                    if base_variant.endswith(suffix):
                        base_variant = base_variant[: -len(suffix)]
                        break
                score_key = f"{model_id}/{base_variant}"
            else:
                results_dir = project_store.model_results_dir(name, rec_key, pass_num=pass_num)
                score_key = rec_key
            imgs = list_output_images(results_dir) if results_dir.exists() else []
            groups.append({
                "label": rec_key,
                "images": [_media_url(name, p) for p in imgs],
                "scores": score_by_key.get(score_key),
            })

    elif state.phase == Phase.STYLE_REFINE:
        pass_num = state.current_model_select_pass or 1
        round_num = state.current_round
        try:
            evaluation = project_store.load_round_evaluation(name, round_num, pass_num=pass_num)
        except FileNotFoundError:
            evaluation = None
        records = project_store.load_all_round_task_records(name, round_num, pass_num=pass_num)
        score_by_model = {}
        if evaluation is not None:
            score_by_model = {
                e.model: {"total": e.total, **e.scores.model_dump()}
                for e in evaluation.evaluations
            }
        for mid in sorted(records):
            results_dir = project_store.round_results_dir(name, round_num, mid, pass_num=pass_num)
            imgs = list_output_images(results_dir) if results_dir.exists() else []
            groups.append({
                "label": mid,
                "images": [_media_url(name, p) for p in imgs],
                "scores": score_by_model.get(mid),
            })

    elif state.phase == Phase.BATCH_T2I:
        batch_num = state.current_batch
        try:
            batch_config = project_store.load_batch_config(name, batch_num)
        except FileNotFoundError:
            batch_config = None
        if batch_config is not None:
            for case in batch_config.cases:
                case_dir = project_store.batch_t2i_case_dir(name, batch_num, case.id)
                imgs = list_output_images(case_dir) if case_dir.exists() else []
                groups.append({
                    "label": f"{case.id} · {case.category}",
                    "images": [_media_url(name, p) for p in imgs],
                    "scores": None,
                })

    elif state.phase == Phase.BATCH_I2I:
        batch_num = state.current_batch
        uploads = project_store.load_i2i_uploads(name, batch_num)
        for i, _upload in enumerate(uploads, 1):
            case_id = f"i2i-{i:03d}"
            case_dir = project_store.batch_i2i_case_dir(name, batch_num, case_id)
            imgs = list_output_images(case_dir) if case_dir.exists() else []
            groups.append({
                "label": case_id,
                "images": [_media_url(name, p) for p in imgs],
                "scores": None,
            })

    return {"phase": state.phase.value, "ref_images": refs, "groups": groups}
```

- [ ] **Step 4: 写 gallery + media 端点**

在 `src/styleclaw/web/routes_projects.py` 顶部 import 区加：

```python
from pathlib import Path
from fastapi.responses import FileResponse
from styleclaw.web.gallery import build_gallery
```

在文件末尾追加：

```python
@router.get("/{name}/gallery")
async def project_gallery(name: str) -> dict:
    try:
        return build_gallery(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"project '{name}' not found")


media_router = APIRouter(prefix="/media", tags=["media"])


def safe_media_path(name: str, file_path: str) -> Path | None:
    """Resolve a media path under the project dir, or None if it escapes.

    Extracted (not inlined in the endpoint) so the traversal guard can be
    unit-tested directly — an httpx/Starlette client normalizes `../` out of
    the URL before it reaches the route, so an endpoint-level test can't
    exercise this branch.
    """
    base = project_store.project_dir(name).resolve()
    target = (base / file_path).resolve()
    if not target.is_relative_to(base):
        return None
    return target


@media_router.get("/{name}/{file_path:path}")
async def media(name: str, file_path: str) -> FileResponse:
    target = safe_media_path(name, file_path)
    if target is None:
        raise HTTPException(status_code=400, detail="invalid path")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(target)
```

在 `src/styleclaw/web/app.py` 的 `create_app()` 里，注册 projects_router 之后加：

```python
    from styleclaw.web.routes_projects import media_router
    app.include_router(media_router)
```

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run python -m pytest tests/web/test_gallery.py -v`
Expected: PASS（5 个测试）

- [ ] **Step 6: 提交**

```bash
git add src/styleclaw/web/gallery.py src/styleclaw/web/routes_projects.py src/styleclaw/web/app.py tests/web/test_gallery.py
git commit -m "feat(web): per-phase gallery + media file serving"
```

---

## Task 9: 计划预览端点（NL → ActionPlan）

**Files:**
- Create: `src/styleclaw/web/routes_runs.py`
- Modify: `src/styleclaw/web/app.py`
- Test: `tests/web/test_routes_plan.py`

- [ ] **Step 1: 写失败测试**

Create `tests/web/test_routes_plan.py`:

```python
import pytest

from styleclaw.core.models import Action, ActionPlan, Phase, ProjectConfig, ProjectState
from styleclaw.storage import project_store


@pytest.fixture
def planned_project(data_root):
    project_store.create_project(
        ProjectConfig(name="p", ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state("p", ProjectState(phase=Phase.STYLE_REFINE, current_round=1))
    return "p"


def test_plan_endpoint(client, planned_project, monkeypatch):
    fake_plan = ActionPlan(
        summary="精炼一轮",
        steps=[Action(name="refine", description="refine", args={})],
        loop=None,
        stop_summary="停在评分后",
    )

    async def fake_plan_fn(llm, project, intent):
        return fake_plan

    monkeypatch.setattr("styleclaw.web.routes_runs.plan", fake_plan_fn)
    # avoid building a real RoleRouter
    monkeypatch.setattr(
        "styleclaw.web.routes_runs._planner_llm",
        lambda: object(),
    )

    resp = client.post("/api/projects/p/plan", json={"intent": "帮我精炼一轮"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"] == "精炼一轮"
    assert body["steps"][0]["name"] == "refine"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/web/test_routes_plan.py -v`
Expected: FAIL — 404（路由未注册）

- [ ] **Step 3: 写实现**

Create `src/styleclaw/web/routes_runs.py`:

```python
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from styleclaw.orchestrator.planner import plan

router = APIRouter(prefix="/api/projects", tags=["runs"])


class PlanRequest(BaseModel):
    intent: str


def _planner_llm():
    """Build an LLM for the planner role. Isolated for easy test override."""
    from styleclaw.core.llm_routing import Role, RoleRouter

    router_obj = RoleRouter.from_env()
    return router_obj.get(Role.PLANNER)


@router.post("/{name}/plan")
async def preview_plan(name: str, req: PlanRequest) -> dict:
    from styleclaw.storage import project_store

    try:
        project_store.load_state(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"project '{name}' not found")
    llm = _planner_llm()
    action_plan = await plan(llm, name, req.intent)
    return action_plan.model_dump()
```

在 `src/styleclaw/web/app.py` 注册：

```python
    from styleclaw.web.routes_runs import router as runs_router
    app.include_router(runs_router)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run python -m pytest tests/web/test_routes_plan.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/styleclaw/web/routes_runs.py src/styleclaw/web/app.py tests/web/test_routes_plan.py
git commit -m "feat(web): NL intent -> ActionPlan preview endpoint"
```

---

## Task 10: 运行启动 + 运行状态端点

**Files:**
- Modify: `src/styleclaw/web/routes_runs.py`
- Modify: `src/styleclaw/web/app.py`
- Test: `tests/web/test_routes_run.py`

run_manager 实例需在 app 内单例共享。做法：存在 `app.state.run_manager`，路由通过 `request.app.state.run_manager` 取。

> **执行者先做这件事**：本任务的 `test_run_single_action_and_poll` 是整个事件流验证的基石——同步 `TestClient` 轮询 `GET /runs/{id}`，而 `asyncio.create_task` 后台运行在请求间推进。这**能**工作，但前提是 conftest 用 `with TestClient(app) as c:` 形式（它让 anyio portal 的事件循环在工作线程里跨请求存活）。**先单独跑这个测试**（`pytest tests/web/test_routes_run.py::test_run_single_action_and_poll -v`）确认它能到达 `done`。若它挂起/永不完成，不要调 sleep——把这些测试改用 `httpx.AsyncClient(transport=httpx.ASGITransport(app=app))` + `@pytest.mark.asyncio`，自己掌控事件循环。先知道这个后备方案，免得在 Task 11 叠了 WS 之后才发现。

- [ ] **Step 1: 写失败测试**

Create `tests/web/test_routes_run.py`:

```python
import time

from styleclaw.core.models import Phase, ProjectConfig, ProjectState
from styleclaw.storage import project_store


def _refine_project(name: str) -> None:
    project_store.create_project(
        ProjectConfig(name=name, ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state(name, ProjectState(phase=Phase.STYLE_REFINE, current_round=1))


def test_run_single_action_and_poll(client, data_root):
    _refine_project("p")
    resp = client.post(
        "/api/projects/p/run",
        json={"steps": [{"name": "approve", "args": {"target": "batch-t2i"}}]},
    )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    status = None
    for _ in range(100):
        snap = client.get(f"/api/projects/p/runs/{run_id}").json()
        status = snap["status"]
        if status in ("done", "error"):
            break
        time.sleep(0.02)
    assert status == "done"
    types = [e["type"] for e in snap["events"]]
    assert types[-1] == "done"
    assert project_store.load_state("p").phase == Phase.BATCH_T2I


def test_run_unknown_run_id_404(client, data_root):
    _refine_project("p")
    resp = client.get("/api/projects/p/runs/nonexistent")
    assert resp.status_code == 404


def test_run_rejects_empty_steps(client, data_root):
    _refine_project("p")
    resp = client.post("/api/projects/p/run", json={"steps": []})
    assert resp.status_code == 400
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/web/test_routes_run.py -v`
Expected: FAIL — 404/405（端点未实现）

- [ ] **Step 3: 写实现**

在 `src/styleclaw/web/routes_runs.py` 顶部 import 区补充：

```python
from typing import Any
from fastapi import Request
from styleclaw.core.models import Action, ActionPlan, LoopConfig
from styleclaw.orchestrator.actions import ACTION_REGISTRY
from styleclaw.web.run_manager import RunConflict
```

追加请求模型与端点：

```python
class StepIn(BaseModel):
    name: str
    description: str = ""
    args: dict[str, Any] = {}


class RunRequest(BaseModel):
    steps: list[StepIn]
    loop: dict[str, Any] | None = None
    summary: str = ""
    stop_summary: str = ""


@router.post("/{name}/run")
async def start_run(name: str, req: RunRequest, request: Request) -> dict:
    from styleclaw.storage import project_store

    try:
        project_store.load_state(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"project '{name}' not found")
    if not req.steps:
        raise HTTPException(status_code=400, detail="steps must not be empty")
    for s in req.steps:
        if s.name not in ACTION_REGISTRY:
            raise HTTPException(status_code=400, detail=f"unknown action: {s.name}")
        if ACTION_REGISTRY[s.name].requires_confirmation:
            raise HTTPException(
                status_code=400,
                detail=f"action '{s.name}' requires a dedicated confirm endpoint, not /run",
            )
    action_plan = ActionPlan(
        summary=req.summary or req.steps[0].name,
        steps=[Action(name=s.name, description=s.description or s.name, args=s.args) for s in req.steps],
        loop=LoopConfig(**req.loop) if req.loop else None,
        stop_summary=req.stop_summary,
    )
    mgr = request.app.state.run_manager
    kind = "plan" if len(req.steps) > 1 or action_plan.loop else "action"
    try:
        run_id = await mgr.start(name, action_plan, kind=kind)
    except RunConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"run_id": run_id}


@router.get("/{name}/runs/{run_id}")
async def get_run(name: str, run_id: str, request: Request) -> dict:
    mgr = request.app.state.run_manager
    try:
        return mgr.get(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
```

在 `src/styleclaw/web/app.py` 的 `create_app()` 里，创建 app 后、注册路由前加：

```python
    from styleclaw.web.run_manager import RunManager
    app.state.run_manager = RunManager()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run python -m pytest tests/web/test_routes_run.py -v`
Expected: PASS（3 个测试）

- [ ] **Step 5: 提交**

```bash
git add src/styleclaw/web/routes_runs.py src/styleclaw/web/app.py tests/web/test_routes_run.py
git commit -m "feat(web): run start + run status endpoints via RunManager"
```

---

## Task 11: WebSocket 事件流

**Files:**
- Modify: `src/styleclaw/web/routes_runs.py`
- Test: `tests/web/test_ws.py`

- [ ] **Step 1: 写失败测试**

Create `tests/web/test_ws.py`:

```python
import time

from styleclaw.core.models import Phase, ProjectConfig, ProjectState
from styleclaw.storage import project_store


def _refine_project(name: str) -> None:
    project_store.create_project(
        ProjectConfig(name=name, ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state(name, ProjectState(phase=Phase.STYLE_REFINE, current_round=1))


def test_ws_streams_until_done(client, data_root):
    _refine_project("p")
    run_id = client.post(
        "/api/projects/p/run",
        json={"steps": [{"name": "approve", "args": {"target": "batch-t2i"}}]},
    ).json()["run_id"]

    received = []
    with client.websocket_connect(f"/api/projects/p/events?run_id={run_id}") as ws:
        for _ in range(50):
            ev = ws.receive_json()
            received.append(ev["type"])
            if ev["type"] in ("done", "error"):
                break
    assert "done" in received


def test_ws_no_run_closes(client, data_root):
    _refine_project("p")
    # no active run, no run_id -> server should close promptly
    with client.websocket_connect("/api/projects/p/events") as ws:
        ev = ws.receive_json()
        assert ev["type"] == "error"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/web/test_ws.py -v`
Expected: FAIL — WS 路由不存在（连接被拒）

- [ ] **Step 3: 写实现**

在 `src/styleclaw/web/routes_runs.py` 顶部 import 区补：

```python
import asyncio
from fastapi import WebSocket
```

追加 WS 端点（注意：WebSocket 路由也挂在同一 `router` 上，前缀 `/api/projects`）：

```python
@router.websocket("/{name}/events")
async def ws_events(websocket: WebSocket, name: str) -> None:
    await websocket.accept()
    mgr = websocket.app.state.run_manager
    run_id = websocket.query_params.get("run_id") or mgr.active_run_id(name)
    if not run_id:
        await websocket.send_json(
            {"type": "error", "message": "no active run for project", "detail": ""}
        )
        await websocket.close()
        return
    try:
        queue, replay = mgr.subscribe(run_id)
    except KeyError:
        await websocket.send_json(
            {"type": "error", "message": f"run '{run_id}' not found", "detail": ""}
        )
        await websocket.close()
        return
    try:
        for ev in replay:
            await websocket.send_json(ev)
            if ev["type"] in ("done", "error"):
                await websocket.close()
                return
        while True:
            ev = await queue.get()
            await websocket.send_json(ev)
            if ev["type"] in ("done", "error"):
                break
    except Exception:  # noqa: BLE001 - client disconnect etc.
        pass
    finally:
        mgr.unsubscribe(run_id, queue)
        try:
            await websocket.close()
        except RuntimeError:
            pass
```

> **执行者注意**：`run` 在后台 task 跑，事件被缓冲。若运行在 WS 连接前已结束，`replay` 会补齐全部事件（含 `done`），上面的循环里命中 `done` 即关闭——所以测试无竞态。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run python -m pytest tests/web/test_ws.py -v`
Expected: PASS（2 个测试）

- [ ] **Step 5: 提交**

```bash
git add src/styleclaw/web/routes_runs.py tests/web/test_ws.py
git commit -m "feat(web): WebSocket event stream with replay"
```

---

## Task 12: 确认动作端点（init / select-model / add-refs）

**Files:**
- Create: `src/styleclaw/web/routes_confirm.py`
- Modify: `src/styleclaw/web/app.py`
- Test: `tests/web/test_routes_confirm.py`

这三个动作走独立端点，参数由请求直接给齐，**同步**执行（经 `build_context` + `execute` 单步计划，复用确认旁路：不传 `on_confirm`，`requires_confirmation` 分支因 `on_confirm=None` 直接跳过校验、按已给 args 执行）。`init`/`add-refs` 接收 multipart 上传，落到临时目录后把目录路径喂给对应 action 的 `ref_dir`/`image_dir`。

- [ ] **Step 1: 写失败测试**

Create `tests/web/test_routes_confirm.py`:

```python
import io

from styleclaw.core.models import Phase, ProjectConfig, ProjectState
from styleclaw.storage import project_store


def _png_bytes() -> bytes:
    # 1x1 PNG is enough for init's extension-based discovery (no decode here).
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
        b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def test_select_model_advances_phase(client, data_root):
    project_store.create_project(
        ProjectConfig(name="p", ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state("p", ProjectState(phase=Phase.MODEL_SELECT, current_model_select_pass=1))
    resp = client.post(
        "/api/projects/p/select-model",
        json={"models": "mj-v7", "variant": "prompt-only"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert project_store.load_state("p").phase == Phase.STYLE_REFINE


def test_select_model_rejects_unknown(client, data_root):
    project_store.create_project(
        ProjectConfig(name="p", ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state("p", ProjectState(phase=Phase.MODEL_SELECT, current_model_select_pass=1))
    resp = client.post("/api/projects/p/select-model", json={"models": "no-such-model"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is False


def test_init_creates_project(client, data_root, monkeypatch):
    # Stub the RunningHub upload that init performs so no network is needed.
    async def fake_init_project(name, refs, ip_info, description, client_, force=False):
        from styleclaw.core.models import ProjectConfig as PC
        project_store.create_project(
            PC(name=name, ip_info=ip_info, ref_images=[f"refs/{p.name}" for p in refs]),
            force=force,
        )
        return project_store.project_dir(name)

    monkeypatch.setattr(
        "styleclaw.scripts.init_project.init_project", fake_init_project
    )

    files = [("files", ("ref-001.png", io.BytesIO(_png_bytes()), "image/png"))]
    resp = client.post(
        "/api/projects",
        data={"name": "newproj", "ip_info": "anime", "description": "d"},
        files=files,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["ok"] is True
    assert "newproj" in project_store.list_projects()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/web/test_routes_confirm.py -v`
Expected: FAIL — 404/405（端点未实现）

- [ ] **Step 3: 写实现**

Create `src/styleclaw/web/routes_confirm.py`:

```python
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from styleclaw.core.models import Action, ActionPlan
from styleclaw.orchestrator.actions import ACTION_REGISTRY, StepResult
from styleclaw.orchestrator.executor import execute
from styleclaw.web.context import build_context

router = APIRouter(prefix="/api/projects", tags=["confirm"])

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


async def _run_single(project: str, action: str, args: dict) -> StepResult:
    action_def = ACTION_REGISTRY[action]
    plan = ActionPlan(
        summary=action,
        steps=[Action(name=action, description=action, args=args)],
        loop=None,
        stop_summary="",
    )
    async with build_context(
        project, needs_client=action_def.needs_client, needs_llm=action_def.needs_llm,
    ) as ctx:
        # No on_confirm passed: the requires_confirmation branch is skipped and
        # the action runs with the args we supply directly.
        results = await execute(plan, ctx)
    return results[-1] if results else StepResult(ok=False, message="no result")


def _result_payload(result: StepResult) -> dict:
    return {"ok": result.ok, "message": result.message, "data": result.data}


async def _save_uploads(files: list[UploadFile]) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="styleclaw-upload-"))
    saved = 0
    for f in files:
        fname = Path(f.filename or "").name
        if not fname or Path(fname).suffix.lower() not in _IMAGE_EXTS:
            continue
        dest = tmp_dir / fname
        dest.write_bytes(await f.read())
        saved += 1
    if saved == 0:
        raise HTTPException(status_code=400, detail="no valid image files uploaded")
    return tmp_dir


@router.post("")
async def create_project(
    name: str = Form(...),
    ip_info: str = Form(""),
    description: str = Form(""),
    force: bool = Form(False),
    files: list[UploadFile] = File(...),
) -> dict:
    tmp_dir = await _save_uploads(files)
    result = await _run_single(
        name, "init",
        {"ref_dir": str(tmp_dir), "ip_info": ip_info, "description": description, "force": force},
    )
    return _result_payload(result)


class SelectModelRequest(BaseModel):
    models: str
    variant: str = ""


@router.post("/{name}/select-model")
async def select_model(name: str, req: SelectModelRequest) -> dict:
    result = await _run_single(
        name, "select-model", {"models": req.models, "variant": req.variant},
    )
    return _result_payload(result)


@router.post("/{name}/refs")
async def add_refs(name: str, files: list[UploadFile] = File(...)) -> dict:
    tmp_dir = await _save_uploads(files)
    result = await _run_single(name, "add-refs", {"image_dir": str(tmp_dir)})
    return _result_payload(result)
```

在 `src/styleclaw/web/app.py` 注册（放在 projects_router 之前，确保 `POST /api/projects` 由 confirm 路由处理而非与 list 冲突——FastAPI 按方法区分，GET/POST 不冲突，顺序不影响）：

```python
    from styleclaw.web.routes_confirm import router as confirm_router
    app.include_router(confirm_router)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run python -m pytest tests/web/test_routes_confirm.py -v`
Expected: PASS（3 个测试）

- [ ] **Step 5: 提交**

```bash
git add src/styleclaw/web/routes_confirm.py src/styleclaw/web/app.py tests/web/test_routes_confirm.py
git commit -m "feat(web): confirm-action endpoints (init/select-model/add-refs)"
```

---

## Task 13: CLI `styleclaw web` 启动命令

**Files:**
- Create: `src/styleclaw/web/launch.py`
- Modify: `src/styleclaw/cli.py`
- Test: `tests/web/test_launch.py`

- [ ] **Step 1: 写失败测试**

Create `tests/web/test_launch.py`:

```python
from styleclaw.web.launch import build_server_config


def test_build_server_config_defaults():
    cfg = build_server_config(port=8800, open_browser=False)
    assert cfg["host"] == "127.0.0.1"
    assert cfg["port"] == 8800


def test_web_command_registered():
    from typer.testing import CliRunner
    from styleclaw.cli import app

    result = CliRunner().invoke(app, ["web", "--help"])
    assert result.exit_code == 0
    assert "port" in result.output.lower()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/web/test_launch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'styleclaw.web.launch'`

- [ ] **Step 3: 写实现**

Create `src/styleclaw/web/launch.py`:

```python
from __future__ import annotations

import logging
import threading
import webbrowser

logger = logging.getLogger(__name__)

_HOST = "127.0.0.1"


def build_server_config(port: int = 8800, open_browser: bool = True) -> dict:
    """Return the uvicorn config dict. Isolated so tests can assert it without
    actually binding a socket."""
    return {"host": _HOST, "port": port, "open_browser": open_browser}


def serve(port: int = 8800, open_browser: bool = True) -> None:  # pragma: no cover - runs a real server
    import uvicorn

    from styleclaw.web.app import create_app

    cfg = build_server_config(port=port, open_browser=open_browser)
    if cfg["open_browser"]:
        url = f"http://{cfg['host']}:{cfg['port']}"
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        logger.info("Opening browser at %s", url)
    uvicorn.run(create_app(), host=cfg["host"], port=cfg["port"], log_level="info")
```

在 `src/styleclaw/cli.py` 末尾（任意 `@app.command()` 之后）加：

```python
@app.command()
def web(
    port: int = typer.Option(8800, help="Port to serve the web UI on"),
    open_browser: bool = typer.Option(
        True, "--open-browser/--no-open-browser", help="Auto-open the browser",
    ),
) -> None:
    """Launch the local web UI (single-user, binds to 127.0.0.1)."""
    from styleclaw.web.launch import serve

    serve(port=port, open_browser=open_browser)
```

并把 `web` 加入 `_global_options` 的 `_skip_validation` 集合（web 启动本身不需要 key——缺 key 时由前端提示页处理，符合 spec §5.5）。在 `cli.py:48-51` 的集合里加 `"web"`：

```python
    _skip_validation = {
        "status", "rollback", "set-sref", "set-pass", "migrate",
        "archive", "clean", "web",
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run python -m pytest tests/web/test_launch.py -v`
Expected: PASS（2 个测试）

- [ ] **Step 5: 提交**

```bash
git add src/styleclaw/web/launch.py src/styleclaw/cli.py tests/web/test_launch.py
git commit -m "feat(cli): styleclaw web command to launch the local server"
```

---

## Task 14: 端到端冒烟 + 全量回归

**Files:**
- Test: `tests/web/test_smoke_e2e.py`

- [ ] **Step 1: 写端到端冒烟测试**

Create `tests/web/test_smoke_e2e.py`:

```python
import time

from styleclaw.core.models import Phase, ProjectConfig, ProjectState
from styleclaw.storage import project_store


def test_full_run_then_gallery(client, data_root):
    # Park a project in STYLE_REFINE, run `approve` via /run, watch WS to done,
    # then read the gallery for the new phase. Exercises the whole M1 spine
    # with no network (approve needs neither client nor llm).
    project_store.create_project(
        ProjectConfig(name="p", ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state("p", ProjectState(phase=Phase.STYLE_REFINE, current_round=1))

    run_id = client.post(
        "/api/projects/p/run",
        json={"steps": [{"name": "approve", "args": {"target": "batch-t2i"}}]},
    ).json()["run_id"]

    seen = []
    with client.websocket_connect(f"/api/projects/p/events?run_id={run_id}") as ws:
        for _ in range(50):
            ev = ws.receive_json()
            seen.append(ev["type"])
            if ev["type"] in ("done", "error"):
                break
    assert seen[-1] == "done"

    detail = client.get("/api/projects/p").json()
    assert detail["state"]["phase"] == "BATCH_T2I"

    gallery = client.get("/api/projects/p/gallery").json()
    assert gallery["phase"] == "BATCH_T2I"
```

- [ ] **Step 2: 跑端到端测试**

Run: `uv run python -m pytest tests/web/test_smoke_e2e.py -v`
Expected: PASS

- [ ] **Step 3: 全量回归 + 覆盖率门槛**

Run: `uv run python -m pytest tests/ -q`
Expected: 全部通过；新增 web 测试不破坏既有测试。

Run: `uv run python -m pytest tests/ --cov=src`
Expected: 总覆盖率 ≥ 80%（`fail_under=80`）。若 web 模块拉低覆盖率，补针对未覆盖分支（如 gallery 的 BATCH_I2I、media traversal）的小测试。

- [ ] **Step 4: 手动验证（可选，非自动化）**

```bash
uv run styleclaw web --no-open-browser --port 8800 &
curl -s http://127.0.0.1:8800/api/health
curl -s http://127.0.0.1:8800/api/projects
# websocat ws://127.0.0.1:8800/api/projects/<name>/events?run_id=<id>
kill %1
```

- [ ] **Step 5: 提交**

```bash
git add tests/web/test_smoke_e2e.py
git commit -m "test(web): end-to-end smoke for run->events->gallery spine"
```

---

## 自检（写计划者已执行）

**1. Spec 覆盖：**
- §1.2 本地单用户 / 127.0.0.1 / 无鉴权 → Task 13（绑 127.0.0.1）、全程无 auth ✓
- §1.6 浏览器内完整生命周期 → 项目 CRUD（Task 7/12）、各阶段动作经 /run（Task 10）、图库（Task 8）✓
- §1.6 聊天 NL→plan→预览 → Task 9（预览）+ Task 10（执行 plan）✓
- §1.6 token 逐字流式 → Task 4/5/6（sink + 接线 + run_manager 发 llm_delta）✓
- §1.6 断线重连 → Task 6（事件缓冲）+ Task 10（runs/{id} 重放）+ Task 11（WS replay）✓
- §3.1 确认动作走专属端点不进循环 → Task 12 + Task 10（/run 拒绝 requires_confirmation）✓
- §4.1 各 REST 端点 → Task 7-12 ✓
- §4.2 事件 schema → Task 3 ✓
- §5.2 panel/并发回退 → Task 6 `_panel_active()` 守卫 ✓
- §5.5 key 缺失 → Task 13（web 跳过 env 校验，留给前端提示）✓（前端提示页属 M2）
- §9 测试策略（TestClient + tmp DATA_ROOT + stub）→ 各 Task 测试 ✓
- §10 里程碑 M1 = 可 curl/websocat 验证的纯后端 → 本计划即 M1，Task 14 手动验证 ✓

**M1 不含（属 M2/M3，spec 已划定）**：React 前端、`/` 与 `/assets/*` 静态托管、前端提示页、成本前置确认 UI（`cost_estimate` 已存在，UI 在 M2 接）、双击启动脚本、token 流式的并发路径。

**2. 占位符扫描：** 无 TBD/TODO；每个改代码的步骤都给了完整代码块。

**3. 类型/签名一致性：**
- `RunManager.start(project, plan, *, kind)` / `.get(run_id)` / `.subscribe(run_id)` / `.unsubscribe(run_id, queue)` / `.active_run_id(project)` 在 Task 6 定义，Task 10/11 调用一致 ✓
- `build_context(project, *, needs_client, needs_llm, router)`(Task 2) 在 Task 6/12 调用一致 ✓
- 事件 `type` 字面量（run_started/step_start/llm_delta/step_done/needs_human/phase_paused/done/error）Task 3 定义，Task 6/11 使用一致 ✓
- `emit_delta`/`set_delta_sink`/`reset_delta_sink`(Task 4) 在 Task 5/6 使用一致 ✓
- `ActionPlan`/`Action`/`LoopConfig`/`StepResult` 均来自现有 `core.models`/`orchestrator.actions`，签名以本仓现状为准 ✓

> **遗留给 M2/M3 的接口注记**：`needs_human` / `phase_paused` 两个事件已在 schema 中定义，但 M1 的 `execute()` 钩子只发 step/run 级事件——`phase_paused` 语义在 M1 等价于一次 `/run` 自然结束的 `done`（前端 M2 据此显示「继续」）；`needs_human` 的发射需要在 run_manager 里读取 `_should_continue_loop` 的判定信号，留待 M2 接入带 loop 的多轮 plan 时补一个 `on_step_done` 内的评估读取分支。M1 schema 先就位，保证前后端契约稳定。
