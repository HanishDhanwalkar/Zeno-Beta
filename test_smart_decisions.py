"""Hermetic tests for smart-decision parsing and SDK interaction."""

import json
import unittest
from types import SimpleNamespace

from decision.engine import evaluate_response, evaluate_routing


class SmartDecisionTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.results = {}
        self.timeouts = []

        class FakeChoice:
            def __init__(self, *, instructions, criteria):
                self.instructions = instructions
                self.criteria = criteria

        class FakeClient:
            def __init__(inner_self, *, timeout):
                self.timeouts.append(timeout)

            def __enter__(inner_self):
                return inner_self

            def __exit__(inner_self, exc_type, exc_value, traceback):
                return False

            def system_one(inner_self, *, state, questions):
                self.requests.append({"state": state, "questions": questions})
                return SimpleNamespace(choices=self.results)

        self.client_factory = FakeClient
        self.choice_factory = FakeChoice

    def test_routing_choices_control_history_and_redis_hints(self):
        self.results = {
            "IS_HISTORY_REQUIRED": SimpleNamespace(choice="yes"),
            "IS_REDIS_HITS_REQUIRED": SimpleNamespace(choice="no"),
        }

        decision = evaluate_routing(
            {"prompt": "Do that again?", "history": [{"role": "user", "content": "List files"}]},
            client_factory=self.client_factory,
            choice_factory=self.choice_factory,
            timeout=7,
        )

        self.assertTrue(decision.history_required)
        self.assertFalse(decision.redis_hits_required)
        request = self.requests[0]
        self.assertEqual(self.timeouts, [7])
        self.assertEqual(
            json.loads(request["state"]["body"])["prompt"],
            "Do that again?",
        )
        self.assertEqual(
            set(request["questions"]),
            {"IS_HISTORY_REQUIRED", "IS_REDIS_HITS_REQUIRED"},
        )

    def test_response_decision_requires_both_positive_answers(self):
        self.results = {
            "IS_RESPONSE_ANSWERS_QUERY": SimpleNamespace(choice="yes"),
            "SHOULD_STOP": SimpleNamespace(choice="no"),
        }

        decision = evaluate_response(
            {"prompt": "Explain the result.", "answer": "The result is 42."},
            client_factory=self.client_factory,
            choice_factory=self.choice_factory,
        )

        self.assertTrue(decision.answers_query)
        self.assertFalse(decision.should_stop)
        self.assertEqual(
            set(self.requests[0]["questions"]),
            {"IS_RESPONSE_ANSWERS_QUERY", "SHOULD_STOP"},
        )

    def test_invalid_or_missing_choices_raise_instead_of_defaulting(self):
        self.results = {
            "IS_HISTORY_REQUIRED": SimpleNamespace(choice="maybe"),
            "IS_REDIS_HITS_REQUIRED": SimpleNamespace(choice="no"),
        }

        with self.assertRaisesRegex(ValueError, "invalid choice"):
            evaluate_routing(
                {"prompt": "Hello"},
                client_factory=self.client_factory,
                choice_factory=self.choice_factory,
            )

        self.results = {}
        with self.assertRaisesRegex(ValueError, "invalid choice"):
            evaluate_routing(
                {"prompt": "Hello"},
                client_factory=self.client_factory,
                choice_factory=self.choice_factory,
            )

    def test_invalid_state_and_timeout_are_rejected(self):
        with self.assertRaises(TypeError):
            evaluate_routing(
                [],
                client_factory=self.client_factory,
                choice_factory=self.choice_factory,
            )
        with self.assertRaisesRegex(ValueError, "timeout must be positive"):
            evaluate_routing(
                {"prompt": "Hello"},
                client_factory=self.client_factory,
                choice_factory=self.choice_factory,
                timeout=0,
            )


if __name__ == "__main__":
    unittest.main()
