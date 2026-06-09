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
