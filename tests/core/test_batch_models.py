from styleclaw.core.models import BatchCase, BatchConfig, ProjectState, Phase


class TestBatchCase:
    def test_defaults(self):
        c = BatchCase(id="case-001", category="adult_male", description="test")
        assert c.aspect_ratio == "9:16"
        assert c.status == "pending"

    def test_full_fields(self):
        c = BatchCase(
            id="case-outdoor-01",
            category="outdoor_scene",
            description="A park at sunset",
            aspect_ratio="16:9",
            status="completed",
        )
        assert c.aspect_ratio == "16:9"
        assert c.status == "completed"


class TestBatchConfig:
    def test_defaults(self):
        bc = BatchConfig()
        assert bc.batch == 0
        assert bc.trigger_phrase == ""
        assert bc.cases == []

    def test_with_cases(self):
        cases = [
            BatchCase(id="c1", category="adult_male", description="test1"),
            BatchCase(id="c2", category="adult_female", description="test2"),
        ]
        bc = BatchConfig(batch=1, trigger_phrase="watercolor", cases=cases)
        assert len(bc.cases) == 2
        assert bc.cases[0].id == "c1"


class TestProjectStateWithBatch:
    def test_with_batch(self):
        state = ProjectState(phase=Phase.BATCH_T2I)
        new_state = state.with_batch(3)
        assert new_state.current_batch == 3
        assert state.current_batch == 0
