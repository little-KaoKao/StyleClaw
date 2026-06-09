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
