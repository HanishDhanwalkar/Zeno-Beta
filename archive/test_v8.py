"""Offline tests for v8 session tracing; no Ollama or Redis required."""

import io
import json
import runpy
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch


v8 = runpy.run_path(str(Path(__file__).resolve().parents[1] / "v-8.py"))
SessionLog = v8["SessionLog"]
SessionLogError = v8["SessionLogError"]
ThinkingCancelled = v8["ThinkingCancelled"]


class SessionLogTests(unittest.TestCase):
    def make_log(self):
        with patch.object(SessionLog, "_persist"):
            return SessionLog(Path.cwd())

    def test_events_are_ordered_and_snapshot_mutable_data(self):
        log = self.make_log()
        data = {"content": "hello", "steps": []}
        with patch.object(log, "_persist") as persist:
            log.append("user", data, task_id="task", turn_id="turn")
            data["steps"].append("later")
            log.finish("completed")
        events = log.document["events"]
        self.assertEqual([event["sequence"] for event in events], [1, 2])
        self.assertEqual(events[0]["data"]["steps"], [])
        self.assertEqual(events[0]["turn_id"], "turn")
        self.assertEqual(log.document["status"], "completed")
        self.assertIsNotNone(log.document["ended_at"])
        self.assertEqual(persist.call_count, 2)

    def test_atomic_json_persists_and_failed_replace_preserves_previous_file(self):
        with TemporaryDirectory() as directory:
            log = SessionLog(Path(directory))
            log.append("user", {"prompt": "hello"})
            previous = log.path.read_text(encoding="utf-8")
            self.assertEqual(json.loads(previous)["events"][0]["kind"], "user")
            with patch.object(v8["os"], "replace", side_effect=OSError("disk failure")):
                with self.assertRaisesRegex(SessionLogError, "disk failure"):
                    log.append("model", {"content": "answer"})
            self.assertEqual(log.path.read_text(encoding="utf-8"), previous)
            self.assertEqual(list(log.path.parent.iterdir()), [log.path])

    def test_serialization_error_is_explicit(self):
        log = self.make_log()
        with self.assertRaisesRegex(SessionLogError, "JSON"):
            log.append("bad", {"value": object()})
        self.assertEqual(log.document["events"], [])

    def test_request_formats_are_deduplicated_without_mutating_payload(self):
        log = self.make_log()
        payload = {"format": {"type": "object"}, "messages": []}
        with patch.object(log, "_persist"):
            log.append("model_request", {"payload": payload})
            log.append("model_request", {"payload": payload})
            log.append("model_request", {"payload": {"format": {"type": "string"}}})
        events = log.document["events"]
        self.assertEqual(len(log.document["formats"]), 2)
        self.assertEqual(events[0]["data"]["format_ref"], events[1]["data"]["format_ref"])
        self.assertNotEqual(events[0]["data"]["format_ref"], events[2]["data"]["format_ref"])
        self.assertNotIn("format", events[0]["data"]["payload"])
        self.assertEqual(payload["format"], {"type": "object"})
        for event in events:
            self.assertIn(event["data"]["format_ref"], log.document["formats"])

    def test_failed_request_persistence_does_not_leave_an_orphan_format(self):
        log = self.make_log()
        with patch.object(log, "_persist", side_effect=SessionLogError("disk full")):
            with self.assertRaises(SessionLogError):
                log.append("model_request", {"payload": {"format": {"type": "object"}}})
        self.assertEqual(log.document["formats"], {})
        self.assertEqual(log.document["events"], [])


class OllamaStreamTests(unittest.TestCase):
    def test_stream_chunks_are_combined_with_completion_metadata(self):
        chunks = [
            {"message": {"content": "{"}, "done": False},
            {"message": {"content": '"name"'}, "done": False},
            {
                "message": {"content": ':"finish"}'},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 17,
            },
        ]
        response = io.BytesIO(
            b"".join(json.dumps(chunk).encode() + b"\n" for chunk in chunks)
        )
        self.assertEqual(
            v8["_read_ollama_stream"](response),
            {
                "message": {"content": '{"name":"finish"}'},
                "done_reason": "stop",
                "prompt_eval_count": 17,
            },
        )

    def test_incomplete_stream_is_not_returned_as_a_model_response(self):
        response = io.BytesIO(b'{"message":{"content":"partial"},"done":false}\n')
        with self.assertRaisesRegex(RuntimeError, "ended before generation completed"):
            v8["_read_ollama_stream"](response)


class TracingTests(unittest.TestCase):
    def setUp(self):
        self.saved = []
        self.persistence = patch.object(
            SessionLog, "_persist", side_effect=lambda doc: self.saved.append(deepcopy(doc)),
        )
        self.persistence.start()
        self.addCleanup(self.persistence.stop)
        self.log = SessionLog(Path.cwd())
        self.memory = Mock()
        self.memory.error_type = RuntimeError
        self.memory.embedding_model = "fake-embedding"
        self.memory.retrieve.return_value = []
        self.memory.action.return_value = None
        self.memory.add.return_value = {"id": "cache-record"}
        self.requests = []

    def run_loop(self, responses, **kwargs):
        responses = iter(responses)

        def respond(*args, **request):
            self.requests.append(deepcopy(request["payload"]))
            response = next(responses)
            if isinstance(response, BaseException):
                raise response
            return {"message": {"content": json.dumps(response)}, "done_reason": "stop"}

        with (
            patch.dict(v8["run_tool_loop"].__globals__, {"request_ollama_stream": respond}),
            patch.object(v8["demo"], "execute", return_value={"exit_code": 0, "stdout": "ok", "stderr": ""}) as execute,
            redirect_stdout(io.StringIO()),
        ):
            report = v8["run_tool_loop"](
                Path.cwd(), "Please execute the command.", memory=self.memory,
                session=self.log, max_requests=3, **kwargs,
            )
        return report, execute

    def test_full_trace_includes_model_payload_cache_tool_and_task_result(self):
        report, execute = self.run_loop([
            {"name": "bash", "arguments": {"command": "echo ok"}},
            {"name": "finish", "arguments": {"answer": "Done", "evidence": []}},
        ])
        self.assertIsNone(report["error"])
        execute.assert_called_once()
        events = self.log.document["events"]
        kinds = [event["kind"] for event in events]
        for kind in ("task_started", "user", "model_request", "model_response",
                     "action", "tool", "task", "task_finished", "cache_result"):
            self.assertIn(kind, kinds)
        model_request = next(event for event in events if event["kind"] == "model_request")
        self.assertLessEqual(len(model_request["data"]["payload"]["messages"]), 4)
        self.assertEqual(model_request["turn_id"], report["turn_id"])
        tool = next(event for event in events if event["kind"] == "tool")
        self.assertEqual(tool["data"]["result"]["stdout"], "ok")
        self.assertFalse(tool["data"]["reused"])
        self.assertEqual(events[-1]["data"]["status"], "completed")
        self.memory.close.assert_not_called()

    def test_cached_action_is_logged_without_execution(self):
        self.memory.action.return_value = {
            "at": "earlier", "turn_id": "prior",
            "data": {"outcome": "success", "result": {"exit_code": 0, "stdout": "cached", "stderr": ""}},
        }
        report, execute = self.run_loop([
            {"name": "bash", "arguments": {"command": "echo ok"}},
            {"name": "finish", "arguments": {"answer": "Historical result", "evidence": []}},
        ])
        execute.assert_not_called()
        self.assertTrue(report["calls"][0]["reused"])
        tool = next(e for e in self.log.document["events"] if e["kind"] == "tool")
        self.assertTrue(tool["data"]["reused"])
        self.assertTrue(tool["data"]["historical_result"])

    def test_completed_command_survives_history_eviction_and_is_not_executed_twice(self):
        command = {"name": "bash", "arguments": {"command": "echo ok"}}
        report, execute = self.run_loop([
            command, command,
            {"name": "finish", "arguments": {"answer": "Done", "evidence": []}},
        ])
        self.assertIsNone(report["error"])
        execute.assert_called_once()
        self.assertTrue(report["calls"][1]["reused"])
        self.assertIn('"completed_tools": [{"name": "bash"', self.requests[2]["messages"][-1]["content"])
        self.assertIn('"stdout": "ok"', self.requests[2]["messages"][-1]["content"])

    def test_first_source_page_survives_history_eviction(self):
        with TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "source.txt").write_text("first page\nsecond page\nthird page\n", encoding="utf-8")
            actions = iter([
                {"name": "read", "arguments": {"path": "source.txt", "from_line": line, "to_line": line}}
                for line in (1, 2, 3)
            ] + [{"name": "finish", "arguments": {
                "answer": "All three pages inspected.",
                "evidence": [{"path": "source.txt", "line": 1, "quote": "first page"}],
            }}])
            requests = []

            def respond(*args, **request):
                requests.append(deepcopy(request["payload"]))
                return {"message": {"content": json.dumps(next(actions))}}

            with (
                patch.dict(v8["run_tool_loop"].__globals__, {"request_ollama_stream": respond}),
                redirect_stdout(io.StringIO()),
            ):
                report = v8["run_tool_loop"](
                    cwd, "Inspect three pages of source.txt.", memory=self.memory,
                    session=self.log, max_requests=4,
                )
            self.assertIsNone(report["error"])
            self.assertEqual(len(requests), 4)
            final_content = requests[-1]["messages"][-1]["content"]
            self.assertIn('"source_pages": [', final_content)
            for text in ("first page", "second page", "third page"):
                self.assertIn(text, final_content)
            self.assertTrue(all("format" in request for request in requests))

    def test_model_failure_and_cache_failure_are_logged(self):
        for operation in ("model", "cache"):
            with self.subTest(operation=operation):
                self.log = SessionLog(Path.cwd())
                self.memory.retrieve.side_effect = RuntimeError("cache unavailable") if operation == "cache" else None
                report, execute = self.run_loop([RuntimeError("model unavailable")])
                self.assertIn("unavailable", report["error"])
                execute.assert_not_called()
                kinds = [event["kind"] for event in self.log.document["events"]]
                self.assertIn("model_error" if operation == "model" else "cache_error", kinds)
                self.assertEqual(self.log.document["events"][-1]["data"]["status"], "failed")

    def test_interruption_records_incomplete_task_and_propagates(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop([KeyboardInterrupt()])
        final = self.log.document["events"][-1]
        self.assertEqual(final["kind"], "task_finished")
        self.assertEqual(final["data"]["status"], "interrupted")
        self.assertIsNone(final["data"]["answer"])

    def test_escape_cancels_generation_without_changing_existing_history(self):
        history = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
        ]
        previous_history = deepcopy(history)
        with patch.dict(
            v8["run_tool_loop"].__globals__,
            {"request_ollama_stream": Mock(side_effect=ThinkingCancelled)},
        ):
            with self.assertRaises(ThinkingCancelled):
                v8["run_tool_loop"](
                    Path.cwd(), "A new question", memory=self.memory,
                    session=self.log, history=history,
                )
        self.assertEqual(history, previous_history)
        self.assertNotIn(
            "model_response",
            [event["kind"] for event in self.log.document["events"]],
        )
        final = self.log.document["events"][-1]
        self.assertEqual(final["kind"], "task_finished")
        self.assertEqual(final["data"]["status"], "interrupted")

    def test_log_failure_aborts_before_cache_or_model_work(self):
        with patch.object(SessionLog, "_persist", side_effect=SessionLogError("disk full")):
            with self.assertRaisesRegex(SessionLogError, "disk full"):
                self.run_loop([])
        self.memory.add.assert_not_called()
        self.memory.retrieve.assert_not_called()

    def test_cli_logs_startup_failure_and_clean_exit(self):
        for failure in (None, RuntimeError("startup unavailable")):
            with self.subTest(failure=failure):
                with (
                    patch("sys.argv", ["v-8.py"]),
                    patch.object(v8["utils"], "check_ollama", side_effect=failure),
                    patch.dict(v8["main"].__globals__, {"create_memory": Mock(return_value=self.memory)}),
                    patch.object(v8["demo"], "read_input", side_effect=EOFError),
                    redirect_stdout(io.StringIO()),
                ):
                    result = v8["main"]()
                self.assertEqual(result, 1 if failure else 0)
                self.assertEqual(self.saved[-1]["status"], "failed" if failure else "completed")


if __name__ == "__main__":
    unittest.main()
