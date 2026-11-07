"""Hermetic regression tests for the v9 harness; no Redis or Ollama needed."""

import hashlib
import io
import json
import runpy
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch


v9 = runpy.run_path(str(Path(__file__).with_name("v-9.py")))
SessionLog = v9["SessionLog"]
ThinkingCancelled = v9["ThinkingCancelled"]


def listing_arguments(**overrides):
    return {
        "path": ".", "pattern": "*.py", "recursive": True,
        "entry_type": "file", "from_index": 0, "limit": 100, **overrides,
    }


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.persistence = patch.object(SessionLog, "_persist")
        self.persistence.start()
        self.addCleanup(self.persistence.stop)
        self.log = SessionLog(Path.cwd())
        self.memory = Mock()
        self.memory.error_type = RuntimeError
        self.memory.retrieve.return_value = []
        self.memory.action.return_value = None
        self.memory.add.return_value = {"id": "record"}
        self.requests = []

    def run_loop(self, actions, *, result=None, prompt="Please execute the command.", **kwargs):
        actions = iter(actions)

        def respond(*args, **request):
            self.requests.append(deepcopy(request["payload"]))
            action = next(actions)
            if isinstance(action, BaseException):
                raise action
            return {"message": {"content": json.dumps(action)}, "done_reason": "stop"}

        with (
            patch.dict(v9["run_tool_loop"].__globals__, {
                "request_ollama_stream": respond,
                "listing_target": Mock(return_value=Path.cwd() if any(
                    word in prompt.lower() for word in ("files", "directories", "folders")
                ) else None),
            }),
            patch.object(v9["demo"], "execute", return_value=result) as execute,
            patch.object(v9["demo"], "require_approval") as approve,
            redirect_stdout(io.StringIO()),
        ):
            report = v9["run_tool_loop"](
                Path.cwd(), prompt, memory=self.memory, session=self.log,
                max_requests=6, **kwargs,
            )
        return report, execute, approve

    @staticmethod
    def action(name, **arguments):
        return {"name": name, "arguments": arguments}

    def finish(self, answer="Done"):
        return self.action("finish", answer=answer, evidence=[])

    def test_list_again_enumerates_fresh_without_entering_general_loop(self):
        cwd = Path.cwd()
        paths = [str(cwd / f"original{i:03}.py") for i in range(110)]
        scan = Mock(side_effect=lambda *args: list(paths))
        with patch.dict(v9["run_tool_loop"].__globals__, {"listing_paths": scan}):
            first, execute, _ = self.run_loop([
                self.action("list", **listing_arguments()),
            ], prompt="Find python files")
            self.assertIsNone(first["error"])
            self.assertEqual(first["last_listing"], {
                "path": str(cwd), "pattern": "*.py",
                "recursive": True, "entry_type": "file",
            })
            self.assertEqual(len(self.requests), 1)
            self.assertGreater(len(first["calls"]), 1)
            added = str(cwd / "newly_added.py")
            paths.append(added)
            with (
                patch.dict(v9["run_tool_loop"].__globals__, {
                    "_run_agent_loop": Mock(side_effect=AssertionError("General loop entered")),
                    "request_ollama_stream": Mock(side_effect=AssertionError("Model called")),
                }),
                redirect_stdout(io.StringIO()),
            ):
                second = v9["run_tool_loop"](
                    cwd, "list again", memory=self.memory, session=self.log,
                    history=first["messages"], last_listing=first["last_listing"],
                )
        self.assertIsNone(second["error"])
        self.assertEqual(second["requests"], 0)
        self.assertEqual(second["calls"][0]["arguments"]["from_index"], 0)
        self.assertEqual(second["last_listing"], first["last_listing"])
        self.assertIn(added, second["answer"])
        self.assertNotIn(added, str(second["messages"]))
        self.assertEqual(
            [path for call in second["calls"] for path in call["result"]["paths"]],
            paths,
        )
        self.assertEqual(scan.call_count, 2)
        self.memory.retrieve.assert_not_called()
        self.memory.action.assert_not_called()
        execute.assert_not_called()

    def test_failed_list_again_preserves_parameters_and_reports_error(self):
        retained = {
            "path": str(Path.cwd()), "pattern": "*.py",
            "recursive": True, "entry_type": "file",
        }
        with (
            patch.dict(v9["run_tool_loop"].__globals__, {
                "listing_paths": Mock(side_effect=PermissionError("access denied")),
                "_run_agent_loop": Mock(side_effect=AssertionError("General loop entered")),
                "request_ollama_stream": Mock(side_effect=AssertionError("Model called")),
            }),
            redirect_stdout(io.StringIO()),
        ):
            report = v9["run_tool_loop"](
                Path.cwd(), "list again", memory=self.memory, session=self.log,
                last_listing=retained,
            )
        self.assertIn("PermissionError", report["error"])
        self.assertIn("Partial listing", report["answer"])
        self.assertEqual(report["last_listing"], retained)
        self.assertEqual(report["requests"], 0)
        self.assertEqual(report["calls"][0]["outcome"], "failed")
        self.assertEqual(self.log.document["events"][-1]["data"]["status"], "failed")

    def test_list_again_without_parameters_reports_explicit_error(self):
        with (
            patch.dict(v9["run_tool_loop"].__globals__, {
                "_run_agent_loop": Mock(side_effect=AssertionError("General loop entered")),
                "request_ollama_stream": Mock(side_effect=AssertionError("Model called")),
            }),
            self.assertRaisesRegex(ValueError, "No previous successful listing"),
        ):
            v9["run_tool_loop"](
                Path.cwd(), "list again", memory=self.memory, session=self.log,
            )
        self.memory.add.assert_not_called()

    def test_truncated_zero_exit_is_partial_and_answer_discloses_it(self):
        report, execute, _ = self.run_loop(
            [self.action("bash", command="echo huge"), self.finish()],
            result={"exit_code": 0, "stdout": "x" * 10000, "stderr": ""},
        )
        self.assertIsNone(report["error"])
        self.assertEqual(report["calls"][0]["outcome"], "partial")
        self.assertIn("partial", report["answer"].lower())
        self.assertIn("incomplete_tools", self.requests[-1]["messages"][-1]["content"])
        execute.assert_called_once()

    def test_duplicate_partial_action_is_corrected_immediately_even_with_retry_reason(self):
        report, execute, _ = self.run_loop([
            self.action("bash", command="echo huge"),
            self.action("bash", command="echo huge", retry_reason="try again"),
            self.finish(),
        ], result={"exit_code": 0, "stdout": "preview", "stderr": "", "truncated": True})
        self.assertEqual([call["outcome"] for call in report["calls"]], ["partial", "blocked"])
        self.assertNotIn("preview", str(report["calls"][1]["result"]))
        self.assertIn("different page", str(report["calls"][1]["result"]))
        execute.assert_called_once()

    def test_historical_oversized_success_is_not_replayed_as_success(self):
        self.memory.action.return_value = {
            "at": "yesterday", "turn_id": "old",
            "data": {"outcome": "success", "result": {"exit_code": 0, "stdout": "x" * 10000}},
        }
        report, execute, _ = self.run_loop([
            self.action("bash", command="echo huge"), self.finish(),
        ])
        self.assertEqual(report["calls"][0]["outcome"], "blocked")
        self.assertIn("partial", report["answer"].lower())
        execute.assert_not_called()

    def test_simple_filtered_shell_listing_uses_bounded_adapter_after_approval(self):
        paths = [str(Path.cwd() / f"file{i:03}.py") for i in range(130)]
        with patch.dict(v9["run_tool_loop"].__globals__, {
            "listing_paths": Mock(return_value=paths),
        }):
            report, execute, approve = self.run_loop([
                self.action("bash", command="Get-ChildItem -Recurse -Filter *.py"),
                self.finish(),
            ])
        result = report["calls"][0]["result"]
        self.assertEqual(result["total_count"], 130)
        self.assertLessEqual(len(result["paths"]), 100)
        self.assertGreater(len(result["paths"]), 0)
        self.assertLessEqual(v9["encoded_size"](result), v9["PAGE_BYTES"])
        self.assertIsNotNone(result["next_list"])
        self.assertIn("partial", report["answer"].lower())
        execute.assert_not_called()
        approve.assert_called_once()

    def test_list_pages_share_snapshot_and_finish_without_partial_disclaimer(self):
        scan = Mock(return_value=[str(Path.cwd() / "a.py"), str(Path.cwd() / "b.py")])
        with patch.dict(v9["run_tool_loop"].__globals__, {"listing_paths": scan}):
            report, execute, approve = self.run_loop([
                self.action("list", path=".", pattern="*.py", recursive=True,
                            entry_type="file", from_index=0, limit=1),
                self.action("list", path=".", pattern="*.py", recursive=True,
                            entry_type="file", from_index=1, limit=1),
                self.finish(),
            ], prompt="List python files")
        self.assertIsNone(report["error"])
        self.assertEqual(report["answer"], "Done")
        self.assertEqual(report["calls"][1]["result"]["next_list"], None)
        scan.assert_called_once()
        execute.assert_not_called()
        approve.assert_not_called()

    def test_completed_directory_listing_requests_only_a_final_answer(self):
        paths = [str(Path.cwd() / name) for name in
                 (".zeno", "__pycache__", "archive", "redis_backend")]
        with patch.dict(v9["run_tool_loop"].__globals__, {
            "listing_paths": Mock(return_value=paths),
        }):
            report, execute, approve = self.run_loop([
                self.action("list", **listing_arguments(pattern="*", recursive=False,
                                                       entry_type="directory", limit=40)),
                self.action("list", **listing_arguments(pattern="*", recursive=False,
                                                       entry_type="directory", from_index=4, limit=40)),
                self.finish("\n".join(paths)),
            ], prompt="can you list directories in this repo")
        self.assertIsNone(report["error"])
        self.assertEqual(len(report["calls"]), 1)
        self.assertEqual([choice["properties"]["name"]["const"]
                          for choice in self.requests[1]["format"]["anyOf"]], ["finish"])
        for path in paths:
            self.assertIn(json.dumps(path), self.requests[-1]["messages"][-1]["content"])
        execute.assert_not_called()
        approve.assert_not_called()

    def test_listing_followed_by_other_work_keeps_tool_actions_available(self):
        with patch.dict(v9["run_tool_loop"].__globals__, {
            "listing_paths": Mock(return_value=[str(Path.cwd() / "a.py")]),
        }):
            report, execute, _ = self.run_loop([
                self.action("list", **listing_arguments()),
                self.action("bash", command="echo next step"),
                self.finish(),
            ], prompt="List python files and execute echo next step",
               result={"exit_code": 0, "stdout": "next step", "stderr": ""})
        self.assertIsNone(report["error"])
        self.assertIn("bash", [choice["properties"]["name"]["const"]
                               for choice in self.requests[1]["format"]["anyOf"]])
        execute.assert_called_once()

    def test_all_listing_pages_survive_history_eviction(self):
        paths = [str(Path.cwd() / f"page{index}.py") for index in range(5)]
        with patch.dict(v9["run_tool_loop"].__globals__, {
            "listing_paths": Mock(return_value=paths),
        }):
            report, _, _ = self.run_loop([
                self.action("list", **listing_arguments(from_index=index, limit=1))
                for index in range(5)
            ] + [self.finish("\n".join(paths))], prompt="List python files")
        self.assertIsNone(report["error"])
        self.assertEqual(report["requests"], 6)
        content = self.requests[-1]["messages"][-1]["content"]
        state = json.loads(content.split("Harness state (data): ", 1)[1])
        retained = [path for entry in state["completed_tools"] for path in entry["result"]["paths"]]
        self.assertEqual(retained, paths)
        self.assertEqual(state["incomplete_tools"], [])

    def test_pending_and_cancelled_exchanges_are_not_model_context(self):
        self.memory.retrieve.return_value = [
            {"id": "pending", "kind": "exchange", "status": "pending"},
            {"id": "cancelled", "kind": "exchange", "status": "cancelled"},
            {"id": "interrupted", "kind": "exchange", "status": "failed",
             "error": "Interrupted: Escape pressed"},
            {"id": "done", "kind": "exchange", "status": "completed",
             "at": "yesterday", "distance": 0.1, "user": "hello", "assistant": "hi"},
        ]
        report, _, _ = self.run_loop([self.finish("Hello")], prompt="hi")
        self.assertIsNone(report["error"])
        content = str(self.requests[0]["messages"])
        self.assertIn('"id": "done"', content)
        for record in ("pending", "cancelled", "interrupted"):
            self.assertNotIn(f'"id": "{record}"', content)

    def test_cancellation_finalizes_memory_without_mutating_history(self):
        history = [{"role": "system", "content": "system"},
                   {"role": "user", "content": "old"},
                   {"role": "assistant", "content": "answer"}]
        previous = deepcopy(history)
        with self.assertRaises(ThinkingCancelled):
            self.run_loop([ThinkingCancelled()], prompt="hi", history=history)
        self.assertEqual(history, previous)
        tasks = [call for call in self.memory.add.call_args_list if call.args[0] == "task"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].args[1]["status"], "cancelled")
        self.assertTrue(tasks[0].args[1]["error"].startswith("Interrupted:"))

    def test_context_and_instruction_cost_are_reduced(self):
        report, _, _ = self.run_loop([self.finish("Hello")], prompt="hi")
        self.assertIsNone(report["error"])
        payload = self.requests[0]
        self.assertEqual(payload["options"]["num_ctx"], 16384)
        self.assertLess(len(payload["messages"][0]["content"]), 3500)
        self.assertEqual(self.log.document["agent_version"], "v9")

    def test_larger_context_remains_an_explicit_override(self):
        report, _, _ = self.run_loop([self.finish("Hello")], prompt="hi", context=32768)
        self.assertIsNone(report["error"])
        self.assertEqual(self.requests[0]["options"]["num_ctx"], 32768)

    def test_successful_commands_still_use_cache_and_failed_commands_can_retry(self):
        for outcome, executions in (("success", 0), ("failed", 1)):
            with self.subTest(outcome=outcome):
                self.memory.action.return_value = {
                    "at": "yesterday", "turn_id": "old",
                    "data": {"outcome": outcome,
                             "result": {"exit_code": 0 if outcome == "success" else 1,
                                        "stdout": "ok", "stderr": ""}},
                }
                report, execute, _ = self.run_loop([
                    self.action("bash", command="echo ok"), self.finish(),
                ], result={"exit_code": 0, "stdout": "ok", "stderr": ""})
                self.assertIsNone(report["error"])
                self.assertEqual(execute.call_count, executions)

    def test_failed_truncated_execution_also_blocks_an_identical_retry(self):
        report, execute, _ = self.run_loop([
            self.action("bash", command="echo huge"),
            self.action("bash", command="echo huge"),
            self.finish(),
        ], result={"exit_code": 1, "stdout": "partial", "truncated": True})
        self.assertEqual([call["outcome"] for call in report["calls"]], ["failed", "blocked"])
        execute.assert_called_once()

    def test_second_duplicate_after_partial_correction_stops_without_more_execution(self):
        action = self.action("bash", command="echo huge")
        report, execute, _ = self.run_loop(
            [action, action, action],
            result={"exit_code": 0, "stdout": "part", "truncated": True},
        )
        self.assertIn("without progress", report["error"])
        self.assertEqual(report["requests"], 3)
        execute.assert_called_once()

    def test_last_listing_page_does_not_hide_a_gap(self):
        with patch.dict(v9["run_tool_loop"].__globals__, {
            "listing_paths": Mock(return_value=["a.py", "b.py", "c.py"]),
        }):
            report, _, _ = self.run_loop([
                self.action("list", **listing_arguments(limit=1)),
                self.action("list", **listing_arguments(from_index=2, limit=1)),
                self.finish(),
            ], prompt="List python files")
        self.assertFalse(report["calls"][-1]["result"]["listing_complete"])
        self.assertIn("partial", report["answer"].lower())
        self.assertIn('"from_index": 1', str(self.requests[-1]["messages"]))

    def test_partial_listing_duplicate_can_be_resolved_with_remaining_pages(self):
        action = self.action("list", **listing_arguments(limit=1))
        with patch.dict(v9["run_tool_loop"].__globals__, {
            "listing_paths": Mock(return_value=["a.py", "b.py"]),
        }):
            report, _, _ = self.run_loop([
                action, action,
                self.action("list", **listing_arguments(from_index=1, limit=1)),
                self.finish(),
            ], prompt="List python files")
        self.assertIsNone(report["error"])
        self.assertEqual(report["answer"], "Done")
        self.assertEqual(report["calls"][1]["outcome"], "blocked")

    def test_complete_listing_can_be_refreshed_with_approval_and_retry_reason(self):
        scan = Mock(side_effect=[["a.py"], ["b.py"]])
        with patch.dict(v9["run_tool_loop"].__globals__, {"listing_paths": scan}):
            report, execute, approve = self.run_loop([
                self.action("bash", command="Get-ChildItem -File -Filter *.py"),
                self.action("bash", command="Get-ChildItem -File -Filter *.py",
                            retry_reason="filesystem changed"),
                self.finish(),
            ])
        self.assertIsNone(report["error"])
        self.assertEqual(scan.call_count, 2)
        self.assertEqual(approve.call_count, 2)
        self.assertEqual(report["calls"][-1]["result"]["paths"], ["b.py"])
        execute.assert_not_called()

    def test_unsupported_listing_commands_still_use_the_normal_shell(self):
        report, execute, approve = self.run_loop([
            self.action("bash", command="Get-ChildItem -Depth 1 | Select-Object -ExpandProperty FullName"),
            self.finish(),
        ], result={"exit_code": 0, "stdout": "a.py", "stderr": ""})
        self.assertIsNone(report["error"])
        execute.assert_called_once()
        approve.assert_not_called()

    def test_declined_listing_is_not_enumerated_or_retried(self):
        with (
            patch.dict(v9["run_tool_loop"].__globals__, {
                "listing_paths": Mock(side_effect=AssertionError("must not enumerate")),
            }),
            patch.object(v9["demo"], "require_approval", side_effect=ValueError("User declined")),
        ):
            actions = iter([
                self.action("bash", command="Get-ChildItem"),
                self.action("bash", command="Get-ChildItem", retry_reason="retry"),
                self.finish(),
            ])
            respond = Mock(side_effect=lambda *a, **k: {
                "message": {"content": json.dumps(next(actions))},
            })
            with (
                patch.dict(v9["run_tool_loop"].__globals__, {"request_ollama_stream": respond}),
                redirect_stdout(io.StringIO()),
            ):
                report = v9["run_tool_loop"](
                    Path.cwd(), "Execute a command", memory=self.memory,
                    session=self.log, max_requests=3,
                )
        self.assertEqual([call["outcome"] for call in report["calls"]], ["failed", "blocked"])

    def test_eof_and_keyboard_interrupt_finalize_exchange(self):
        for exception in (EOFError(), KeyboardInterrupt()):
            with self.subTest(exception=type(exception).__name__):
                self.memory.add.reset_mock()
                with self.assertRaises(type(exception)):
                    self.run_loop([exception])
                task = next(call for call in self.memory.add.call_args_list if call.args[0] == "task")
                self.assertEqual(task.args[1]["status"], "interrupted")
                self.assertTrue(task.args[1]["error"].startswith("Interrupted:"))

    def test_listing_errors_are_explicit_tool_failures(self):
        with patch.dict(v9["run_tool_loop"].__globals__, {
            "listing_paths": Mock(side_effect=PermissionError("access denied")),
        }):
            report, _, _ = self.run_loop([
                self.action("list", **listing_arguments()), self.finish(),
            ], prompt="Inspect a directory")
        self.assertEqual(report["calls"][0]["outcome"], "failed")
        self.assertIn("PermissionError", report["calls"][0]["result"])

    def test_requested_listing_can_explain_a_failure_without_claiming_completion(self):
        with patch.dict(v9["run_tool_loop"].__globals__, {
            "listing_paths": Mock(side_effect=PermissionError("access denied")),
        }):
            report, _, _ = self.run_loop([
                self.action("list", **listing_arguments()), self.finish("Access denied"),
            ], prompt="List python files")
        self.assertIsNone(report["error"])
        self.assertIn("no complete listing", report["answer"])
        self.assertEqual(report["calls"][0]["outcome"], "failed")

    def test_truncated_shell_read_is_resolved_by_complete_checked_read_pages(self):
        cursor = {"path": "source.txt", "from_line": 1, "to_line": 40, "from_column": 0}
        digest = hashlib.sha256(b"hello\n").hexdigest()
        page = {
            "path": "source.txt", "sha256": digest, "total_lines": 1,
            "lines": [{"line": 1, "column": 0, "text": "hello\n"}],
            "next_read": None, "next_line": None, "next_column": 0, "eof": True,
        }
        evidence = [{"path": "source.txt", "line": 1, "quote": "hello"}]
        with (
            patch.dict(v9["run_tool_loop"].__globals__, {
                "shell_file_read": Mock(return_value=cursor),
                "read_page": Mock(return_value=page),
                "read_state": Mock(return_value=digest),
                "file_hash": Mock(return_value=digest),
                "invalidate_source_pages": Mock(return_value=[]),
            }),
            patch.object(Path, "is_dir", return_value=False),
        ):
            report, execute, _ = self.run_loop([
                self.action("bash", command="Get-Content -Raw source.txt"),
                self.action("read", **cursor),
                self.action("finish", answer="Read completely", evidence=evidence),
            ], result={"exit_code": 0, "stdout": "partial", "truncated": True})
        self.assertIsNone(report["error"])
        self.assertEqual(report["answer"], "Read completely")
        self.assertTrue(report["coverage"]["source.txt"])
        execute.assert_called_once()


class ListingTests(unittest.TestCase):
    def test_listing_end_cursor_is_rejected_but_empty_initial_listing_is_valid(self):
        scan = Mock(return_value=["a.py", "b.py"])
        with patch.dict(v9["list_page"].__globals__, {"listing_paths": scan}):
            snapshots = {}
            v9["list_page"](Path.cwd(), listing_arguments(), snapshots)
            with self.assertRaisesRegex(ValueError, "complete"):
                v9["list_page"](Path.cwd(), listing_arguments(from_index=2), snapshots)
            scan.assert_called_once()
        with patch.dict(v9["list_page"].__globals__, {"listing_paths": Mock(return_value=[])}):
            result = v9["list_page"](Path.cwd(), listing_arguments(), {})
        self.assertEqual(result["paths"], [])
        self.assertTrue(result["listing_complete"])

    def test_parser_is_strict_and_retains_scope_filter_and_type(self):
        cwd = Path.cwd()
        parse = v9["shell_listing_arguments"]
        args = parse(cwd, "Get-ChildItem -LiteralPath 'nested folder' -Recurse -File -Filter *.py")
        self.assertEqual(args["path"], "nested folder")
        self.assertEqual(args["pattern"], "*.py")
        self.assertEqual(args["entry_type"], "file")
        self.assertTrue(args["recursive"])
        for command in ("Get-ChildItem | Remove-Item", "Get-ChildItem -Depth 1",
                        "Get-ChildItem -Path $HOME", "Get-ChildItem -Filter",
                        "Get-ChildItem -File -Directory", "echo hello",
                        "Get-ChildItem -Filter $pattern", "Get-ChildItem 'unclosed"):
            with self.subTest(command=command):
                self.assertIsNone(parse(cwd, command))
        self.assertEqual(parse(cwd, "gci 'nested folder' -Name")["path"], "nested folder")

    def test_list_validation_rejects_invalid_pages_types_and_limits(self):
        schema = v9["action_schema"]()
        args = listing_arguments()
        self.assertEqual(v9["validate_action"]({"name": "list", "arguments": args}, schema),
                         ("list", args))
        for key, value in (("limit", 101), ("limit", 0), ("recursive", 1),
                           ("entry_type", "invalid"), ("from_index", -1)):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    v9["validate_action"](
                        {"name": "list", "arguments": {**args, key: value}}, schema,
                    )

    def test_listing_scope_rejects_escape_and_path_globs(self):
        cwd = Path.cwd()
        for overrides in ({"path": ".."}, {"pattern": ""}, {"pattern": r"sub\*.py"},
                          {"pattern": "sub/*.py"}, {"pattern": "a\x00b"}):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    v9["listing_scope"](cwd, listing_arguments(**overrides))

    def test_listing_target_does_not_treat_a_quoted_glob_as_directory(self):
        cwd = Path.cwd()
        with patch.object(Path, "iterdir", return_value=iter([])):
            self.assertEqual(v9["listing_target"](cwd, "List files matching `*.py`"), cwd)

    def test_action_listing_target_covers_adapter_and_ordinary_listings(self):
        cwd = Path.cwd()
        for name, args in (
            ("list", listing_arguments(path="nested")),
            ("bash", {"command": "Get-ChildItem -Path nested -Filter *.py"}),
            ("bash", {"command": "ls nested"}),
        ):
            with self.subTest(name=name, args=args):
                self.assertEqual(v9["action_listing_target"](cwd, name, args), cwd / "nested")
        self.assertIsNone(v9["action_listing_target"](cwd, "finish", {}))

    def test_native_scan_matches_filters_types_recursion_and_sorting(self):
        cwd = Path(r"C:\fake")

        def walk(*args, **kwargs):
            directories = ["sub", "link"]
            yield str(cwd), directories, ["b.py", "a.txt"]
            if "sub" in directories:
                yield str(cwd / "sub"), [], ["deep.py"]

        with (
            patch.object(Path, "resolve", lambda path: path.absolute()),
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "is_symlink", lambda path: path.name == "link"),
            patch.object(v9["os"], "walk", side_effect=walk),
        ):
            for recursive, entry_type, expected in (
                (True, "file", [str(cwd / "b.py"), str(cwd / "sub" / "deep.py")]),
                (False, "file", [str(cwd / "b.py")]),
                (True, "directory", [str(cwd / "link"), str(cwd / "sub")]),
                (True, "all", [str(cwd / "a.txt"), str(cwd / "b.py"),
                               str(cwd / "link"),
                               str(cwd / "sub"), str(cwd / "sub" / "deep.py")]),
            ):
                with self.subTest(recursive=recursive, entry_type=entry_type):
                    args = listing_arguments(
                        recursive=recursive, entry_type=entry_type,
                        pattern="*.py" if entry_type == "file" else "*",
                    )
                    self.assertEqual(v9["listing_paths"](cwd, args), expected)

    def test_scan_errors_are_not_silently_ignored(self):
        cwd = Path.cwd()
        with patch.object(Path, "is_dir", return_value=False):
            with self.assertRaisesRegex(ValueError, "does not exist"):
                v9["listing_paths"](cwd, listing_arguments())

        def fail_walk(*args, **kwargs):
            kwargs["onerror"](PermissionError("denied"))

        with (
            patch.object(Path, "is_dir", return_value=True),
            patch.object(v9["os"], "walk", side_effect=fail_walk),
        ):
            with self.assertRaisesRegex(PermissionError, "denied"):
                v9["listing_paths"](cwd, listing_arguments())

    def test_scan_lists_links_without_following_junctions_or_escaping_workspace(self):
        cwd = Path(r"C:\fake")

        def resolve(path):
            if path.name == "junction":
                return cwd
            if path.name == "outside":
                return cwd.parent / "elsewhere"
            return path.absolute()

        def walk(*args, **kwargs):
            directories = ["junction", "outside", "sub"]
            yield str(cwd), directories, []
            self.assertEqual(directories, ["sub"])
            yield str(cwd / "sub"), [], ["deep.py"]

        with (
            patch.object(Path, "resolve", resolve),
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "is_symlink", return_value=False),
            patch.object(v9["os"], "walk", side_effect=walk),
        ):
            self.assertEqual(
                v9["listing_paths"](cwd, listing_arguments(entry_type="all", pattern="*")),
                [str(cwd / "junction"), str(cwd / "outside"), str(cwd / "sub"),
                 str(cwd / "sub" / "deep.py")],
            )

    def test_page_boundaries_empty_refresh_and_impossibly_large_entry(self):
        namespace = v9["list_page"].__globals__
        scan = Mock(return_value=["a.py", "b.py"])
        cwd = Path.cwd()
        with patch.dict(namespace, {"listing_paths": scan}):
            snapshots = {}
            for invalid in ({"limit": 0}, {"limit": 101}, {"from_index": -1},
                            {"from_index": 3}):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ValueError):
                        v9["list_page"](cwd, listing_arguments(**invalid), snapshots)
            with self.assertRaisesRegex(ValueError, "complete"):
                v9["list_page"](cwd, listing_arguments(from_index=2), snapshots)
            scan.return_value = []
            empty = v9["list_page"](
                cwd, listing_arguments(retry_reason="refresh directory"), snapshots,
            )
            self.assertEqual(empty["total_count"], 0)
            self.assertTrue(empty["listing_complete"])
            scan.return_value = ["x" * 3000]
            with self.assertRaisesRegex(ValueError, "byte budget"):
                v9["list_page"](cwd, listing_arguments(retry_reason="refresh"), snapshots)


if __name__ == "__main__":
    unittest.main()
