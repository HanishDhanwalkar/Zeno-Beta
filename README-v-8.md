# v-8: advisory failures and JSON session traces

[v-8.py](v-8.py) contains the agent loop and independent,
atomically updated JSON session tracing. [v-7.py](v-7.py) has only an appended
docstring failure note; its executable code and Redis backend are unchanged.

```powershell
python .\local_tests\v-8.py --workspace .\local_tests
```

Use the same Redis/Ollama setup as [v-7](README-v-7.md).
Paired-exchange retrieval, `MAX_HISTORY`, source evidence, and approvals remain.

## Task context and repeated actions

`MAX_HISTORY` limits conversation messages, not the current task's verified
source pages or completed tool results. Each model request includes those pages
and successful non-read results alongside the authoritative user task. Earlier
read content therefore remains available after its conversation message is
archived. Changed files still invalidate retained pages.

Identical successful actions within a task reuse their local result, even if
Redis does not return the action. Fresh execution still requires `retry_reason`;
failed attempts remain retryable and current-task declines remain blocked.
The model is instructed to continue from completed steps and finish when done.
Repeated cached requests are still bounded; this cannot guarantee that the model
will never request a redundant action.

If retained task context exceeds the configured context budget, the task stops
with an explicit error rather than silently dropping earlier source content.
Use a smaller task or a larger `--context` in that case.

## Request formats

New session traces use `schema_version: 2`. The top-level `formats` object stores
each distinct response schema once, keyed by its SHA-256 identifier. A
`model_request` event stores `data.format_ref` instead of repeating
`data.payload.format`. Resolve it using `formats[event.data.format_ref]` when
reconstructing the request. Existing session files are not modified.

Ollama still receives `format` on every request: chat requests are independent,
and the schema changes as checked evidence or mandatory recovery cursors change.
This deduplication reduces trace duplication, not model request count or latency.

## Retry policy change

A previous failed action is an observation, not proof that another attempt will
fail. v-7's failure cache blocked a `grep` search before execution and stopped
after repeated cache requests. v-8 permits an identical failed action to execute
again **without `retry_reason`**, in the same task or a later one. Corrected
arguments and alternative methods also remain available.

The terminal shows the previous error in an advisory panel before a fresh
attempt. Normal approval is still required for commands and changes. Declines
in the current task remain blocked. Successful and pending/unknown action
results retain the existing cache policy: fresh execution needs `retry_reason`.
Normal task deadlines and model-request limits still bound unsuccessful loops;
actual failed executions are not counted as repeated cache reuse.

On PowerShell, the model is instructed to use
`Select-String -LiteralPath <file> -Pattern <text> -SimpleMatch` for literal
text searches, rather than assuming `grep` is installed. This is guidance,
not silent rewriting or a ban on other tools.

## Tracing and regression checks

The existing JSON trace records cache lookups, model requests/responses,
pending actions, full tool results, and task outcomes. Retried failures pass
through the same execution/approval path and are recorded as fresh attempts.

[archive/test_v8.py](archive/test_v8.py) contains offline tracing checks plus
regressions for schema deduplication, retained source pages after history
eviction, and successful same-task action reuse without duplicate execution.

Run from this folder with:

```powershell
python -m unittest discover -s archive -p test_v8.py -v
```

Press **Esc** while the model is thinking to stop its streamed response and
return to the prompt. The incomplete turn is not added to active chat history;
earlier conversation history remains available.

The offline suite includes checks for streamed response assembly and history
preservation when generation is cancelled. Tests were not run for this change;
live Ollama/Redis behavior remains unverified.
