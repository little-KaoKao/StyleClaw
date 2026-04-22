from styleclaw.cli import MAX_AUTO_ROUNDS


class TestMaxAutoRounds:
    def test_value_is_five(self):
        assert MAX_AUTO_ROUNDS == 5

    def test_round_exceeds_limit(self):
        current_round = 5
        next_round = current_round + 1
        assert next_round > MAX_AUTO_ROUNDS

    def test_round_within_limit(self):
        current_round = 4
        next_round = current_round + 1
        assert next_round <= MAX_AUTO_ROUNDS
