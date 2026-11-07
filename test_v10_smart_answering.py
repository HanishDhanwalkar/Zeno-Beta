"""Hermetic integration tests for v-10 smart answering."""

import io
import json
import runpy
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


v10 = runpy.run_path(str(Path(__file__).with_name("v-10.py")))
SessionLog = v10["SessionLog"]


class SmartAnsweringIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.persistence = patch.object(SessionLog, "_persist")
        self.persistence.start()
        self.addCleanup(self.persistence.stop)
        self.session = SessionLog(Path.cwd())
        self.memory = Mock()
        self.memory.error_type = RuntimeError
        self.memory.retrieve.return_value = []
        self.memory.action.return_value = None
        self.memory.add.return_value = {"id": "record"}
        self.requests = []

    @staticmethod
    def action(answer):
        return {
            "name": "finish",
            "arguments": {"answer": answer, "evidence": []},
        }

    def run_loop(self, actions, *, history=None, response_check=None):
        actions = iter(actions)

        def respond(*args, **request):
            self.requests.append(request["payload"])
            action = next(actions)
            return {
                "message": {"content": json.dumps(action)},
                "done_reason": "stop",
            }

        routing = Mock(
            return_value=SimpleNamespace(
                history_required=False,
                redis_hits_required=False,
            )
        )
        response_check = response_check or Mock(
            return_value=SimpleNamespace(
                answers_query=True,
                should_stop=True,
            )
        )
        with (
            patch.dict(v10["run_tool_loop"].__globals__, {
                "request_ollama_stream": respond,
                "listing_target": Mock(return_value=None),
                "evaluate_routing": routing,
                "evaluate_response": response_check,
            }),
            redirect_stdout(io.StringIO()),
        ):
            report = v10["run_tool_loop"](
                Path.cwd(),
                "What is the result?",
                history=history,
                memory=self.memory,
                session=self.session,
                max_requests=4,
            )
        return report, routing, response_check

    def test_irrelevant_history_and_redis_hits_are_skipped(self):
        report, routing, response_check = self.run_loop(
            [self.action("The result is 42.")],
            history=[
                {"role": "system", "content": "old system"},
                {"role": "user", "content": "unrelated old request"},
                {"role": "assistant", "content": "unrelated old answer"},
            ],
        )

        self.assertIsNone(report["error"])
        self.assertEqual(report["answer"], "The result is 42.")
        routing.assert_called_once()
        response_check.assert_called_once()
        self.memory.retrieve.assert_not_called()
        self.assertNotIn(
            "unrelated old answer",
            json.dumps(self.requests[0]["messages"]),
        )

    def test_response_rejection_returns_agent_to_the_tool_loop(self):
        rejected = SimpleNamespace(answers_query=False, should_stop=False)
        response_check = Mock(side_effect=[
            rejected,
            SimpleNamespace(answers_query=True, should_stop=True),
        ])
        report, _, _ = self.run_loop(
            [
                self.action("Incomplete answer."),
                self.action("Complete answer."),
            ],
            response_check=response_check,
        )

        self.assertIsNone(report["error"])
        self.assertEqual(report["answer"], "Complete answer.")
        self.assertEqual(response_check.call_count, 2)
        self.assertEqual(len(self.requests), 2)


if __name__ == "__main__":
    unittest.main()
