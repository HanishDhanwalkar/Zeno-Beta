"""Offline checks: python -m unittest test_local_model_demo -v"""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import local_model_demo as demo


class ToolLoopTests(unittest.TestCase):
    def run_loop(self, responses, results=(), max_requests=6):
        requests = []
        responses = iter(responses)

        def respond(url, *, payload, timeout):
            requests.append(deepcopy(payload))
            return {"message": {"content": next(responses)}}

        output = io.StringIO()
        with (
            patch.object(demo, "request_json", side_effect=respond),
            patch.object(demo, "execute", side_effect=results) as execute,
            redirect_stdout(output),
            redirect_stderr(output),
        ):
            report = demo.run_tool_loop(Path.cwd(), "Inspect the file.", "http://local.invalid", 30, max_requests)
        return report, requests, execute, output.getvalue()

    def test_calls_and_results_reach_model_before_final_answer(self):
        read = '{"name":"read","arguments":{"path":"seed.txt"}}'
        finish = '{"name":"finish","arguments":{"answer":"The file says hello."}}'
        report, requests, execute, output = self.run_loop([read, finish], ["hello\n"])
        history = requests[1]["messages"]
        self.assertEqual(history[-2], {"role": "assistant", "content": read})
        result = json.loads(history[-1]["content"].split("\n", 1)[1])
        self.assertEqual(result["result"], "hello\n")
        self.assertEqual(result["outcome"], "success")
        self.assertEqual(result["name"], "read")
        self.assertEqual(report["answer"], "The file says hello.")
        self.assertIsNone(report["error"])
        self.assertEqual(report["requests"], 2)
        execute.assert_called_once()
        self.assertIn("FINAL ANSWER\nThe file says hello.", output)

    def test_failed_tool_result_is_fed_back_for_recovery(self):
        actions = [
            '{"name":"read","arguments":{"path":"missing.txt"}}',
            '{"name":"read","arguments":{"path":"seed.txt"}}',
            '{"name":"finish","arguments":{"answer":"hello"}}',
        ]
        report, requests, execute, output = self.run_loop(
            actions, [FileNotFoundError("missing.txt"), "hello"],
        )
        feedback = requests[1]["messages"][-1]["content"]
        self.assertIn('"outcome": "failed"', feedback)
        self.assertIn("FileNotFoundError", feedback)
        self.assertEqual(report["answer"], "hello")
        self.assertEqual(execute.call_count, 2)

    def test_invalid_action_requests_correction_without_execution(self):
        report, requests, execute, output = self.run_loop([
            '{"name":"read","arguments":{}}',
            '{"name":"finish","arguments":{"answer":"Blocked: no path supplied."}}',
        ])
        self.assertIn("Incorrect argument names", requests[1]["messages"][-1]["content"])
        execute.assert_not_called()
        self.assertIsNone(report["error"])
        self.assertIn("INVALID ACTION", output)

    def test_budget_exhaustion_does_not_invent_final_answer(self):
        report, requests, execute, output = self.run_loop(
            ['{"name":"read","arguments":{"path":"seed.txt"}}'], ["hello"], max_requests=1,
        )
        self.assertIsNone(report["answer"])
        self.assertIn("limit exhausted", report["error"])
        self.assertIn("ERROR", output)
        self.assertNotIn("FINAL ANSWER", output)

    def test_transport_error_is_not_retried_or_hidden(self):
        with (
            patch.object(demo, "request_json", side_effect=RuntimeError("Ollama unavailable")) as request,
            patch.object(demo, "execute") as execute,
            redirect_stderr(io.StringIO()) as errors,
            redirect_stdout(io.StringIO()),
        ):
            report = demo.run_tool_loop(Path.cwd(), "Read.", "http://local.invalid", 30, 2)
        request.assert_called_once()
        execute.assert_not_called()
        self.assertIn("Ollama unavailable", report["error"])
        self.assertIn("Ollama unavailable", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
