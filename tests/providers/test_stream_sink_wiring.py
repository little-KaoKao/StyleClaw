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
