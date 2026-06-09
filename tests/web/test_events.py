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
