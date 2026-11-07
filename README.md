# Local-model tool-loop demos

These standalone scripts use the model and Ollama endpoint from `config.py`.
Start Ollama with that model available, then run:

```powershell
python .\v-5.py --workspace .
```

Enter tasks interactively; `/bye` exits. Commands and file changes require
approval and run with your account permissions, not inside a sandbox.
Earlier versioned scripts are preserved.

## v-10: smart answering

Run `python .\local_tests\v-10.py --workspace .\local_tests`. This keeps the
bounded tool and memory behavior of v-9 while adding local decision checks from
`decision/v-1.py`:

- Before answering, the decision model selects whether prior conversation and
  Redis hints are relevant. Unneeded history and retrieval are omitted.
- Before accepting a final answer, it checks whether the response answers the
  latest request and whether the task is complete. A failed check sends the
  agent back to the existing bounded tool loop.
- Decisions use `typesafe_sdk` with local Ollama defaults
  (`http://localhost:11434`, model `laya`). `TYPESAFE_BASE_URL`,
  `TYPESAFE_API_KEY`, and `TYPESAFE_DEFAULT_MODEL` can override those defaults.
  Decision failures are reported; they do not silently fall back to accepting
  an answer.

`v-9.py` remains unchanged. The decision script can also be exercised directly:

```powershell
python .\local_tests\decision\v-1.py --prompt "We were billed twice for March."
```

The offline smart-answer regression tests need no live Ollama or Redis:

```powershell
python -m unittest discover -s .\local_tests -p "test_*smart*.py"
```

## v-9: bounded output and cancellation-safe recovery

Run `python .\local_tests\v-9.py --workspace .\local_tests` from the repository
root with the same Ollama/Redis setup as v-8. v-8 executable code and shared
backends are unchanged.

- Truncated tool output is **partial**, including commands with exit code zero.
  Identical partial actions receive immediate correction, not another execution
  or the same cached preview. A second duplicate after correction stops the loop.
  Choose a narrower scope, a different page, `read` for source, or `recall` for
  saved historical output. Finishing with unread output adds an explicit partial
  coverage notice.
- The read-only `list` action accepts `path`, `pattern` (for example `*.py`),
  `recursive`, `entry_type` (`all`, `file`, `directory`), `from_index` and `limit`
  (1-100). It returns compact full paths, `total_count`, `next_index` and
  `next_list`. Pages also obey a 2000-byte serialized limit, so long paths can
  reduce the entry count below 100. Follow `next_list` to reuse a task-local
  sorted snapshot without repeating directory enumeration. Complete coverage
  requires every entry, not merely reaching the last page.
  Links/junctions are listed as entries but never traversed recursively.
- For filename-only requests (`list`, `show`, `find`, `locate`, or `enumerate`),
  the model selects listing parameters once using a `list`-only schema. The
  harness follows every page of the snapshot and renders the collected full
  paths and total directly, without another inference request. Negated reads
  such as "dont read files" do not turn a filename request into a reading task.
  Multi-step content/command workflows keep their normal permitted tools.
  Pagination/deadline errors explicitly report partial collected output.
- Successful filename listings retain a small `last_listing` record containing
  only the absolute path, pattern, recursive flag and entry type. The interactive
  session passes it between turns; API callers pass `report["last_listing"]` to
  the next `run_tool_loop` call. "list again" reuses these parameters, resets the
  cursor to zero and enumerates a fresh snapshot through the direct renderer,
  without entering the general agent loop, retrieval, or model inference.
  Full filenames remain accessible in report calls and session tool events,
  never in the structured record or active history. A failed refresh preserves
  the last successful parameters; repeating without prior parameters is an
  explicit error.
- Full listings remain in session/Redis tool events and the returned report,
  not active model history. Multi-step requests receive one aggregate status per
  listing scope rather than every page's paths. Source pages remain available
  for evidence validation outside model state; only current read feedback goes
  into context. Listing requests skip semantic retrieval. Other requests attach
  compact completed, error-free exchange hints without escaped tool previews.
- Explicit prohibitions on reading, writing/editing, shell execution or recall
  filter the response schema before validation/execution. Prohibiting reads also
  removes shell/recall routes that could otherwise expose file contents.
- Input budgeting uses a subword-oriented token estimate over message content
  and schema, calibrated upward against Ollama's actual `prompt_eval_count`.
  Serialized bytes are diagnostic data only, not a token budget. A harness
  estimate rejection explicitly says no request was sent and does not claim
  Ollama exhausted its context. Exact model tokenization remains model-dependent.
- `next_list: null` means stop paging. `next_index` is metadata, not a cursor
  to use after completion. Nonzero indices equal to `total_count` now fail
  explicitly instead of returning repeatable empty pages. Empty directories
  still return a valid complete initial page at index zero.
- Plain `Get-ChildItem`/`gci` listings with a literal path, `-Recurse`, `-Filter`,
  `-File`, `-Directory` or `-Name` are explicitly adapted to the bounded listing
  after approval. Pipelines, scripts and unsupported options still use the
  ordinary approved shell path; they are never silently rewritten.
- Esc/EOF/Ctrl+C finalizes the Redis exchange with an explicit interruption
  error. Pending, cancelled, interrupted and failed exchanges are excluded from model
  hints, including legacy pending records. Existing in-memory history is intact.
  The unchanged backend stores interruptions as failed exchanges with an
  `Interrupted:` error; the task event additionally records cancelled/interrupted
  status, and v-9 filters both representations.
- Default context is reduced from 32768 to **16384** tokens, with shorter system
  instructions and tool feedback. `--context 32768` remains available for larger
  tasks. This lowers configured context/input overhead; model latency is not
  guaranteed. No model, timeout or approval policy was changed.

Offline regression tests (no network or live services):

```powershell
python -m unittest discover -s .\local_tests -p test_v9.py
```

## v-8: advisory historical failures

[v-8](README-v-8.md) retains paired memory and JSON session tracing while
allowing failed actions to execute again without `retry_reason`. Historical
errors are hints, not permanent blocks; approvals and current-task declines
still apply. Successful/unknown cached actions retain explicit fresh retries.
Press Esc while the model is thinking to stop generation without clearing the
existing conversation history.

## v-7: paired user/assistant exchanges

[v-7](README-v-7.md) stores a user query and its assistant response together,
including tool steps, in one incrementally updated searchable Redis record.
Retrieval returns paired hints rather than separate user/assistant hits.
It retains the three-message limit, approvals and exact-action lookup.
Run `python .\local_tests\v-7.py --workspace .\local_tests` from the repository
root after the same Redis/Ollama setup below. v-6 executable code is unchanged.

## v-6: Redis vector memory and deliberate retries

See [the v-6 architecture and sequence diagrams](README-v-6.md) for the current
bounded-context flow, Redis retrieval, exact-action reuse and retry decisions.

[v-6.py](v-6.py) builds on v-5 without modifying it. Its persistent memory
backend is [redis_backend/v-1.py](redis_backend/v-1.py). Use Redis Stack or
Redis 8 with Search / Query Engine enabled; a Redis server without vector
search is insufficient. For the Ubuntu WSL setup, see
[redis_backend/README-v-1.md](redis_backend/README-v-1.md) and start it before
running v-6.

From the repository root:

```powershell
python -m pip install -r .\local_tests\redis_backend\requirements.txt
$env:REDIS_URL = "redis://localhost:6380/0"
$env:ZENO_EMBED_MODEL = "mxbai-embed-large:latest"
python .\local_tests\v-6.py --workspace .\local_tests
```

`REDIS_URL` defaults to the URL above; configure authentication using the
environment, not source code. `ZENO_EMBED_MODEL` defaults to
`mxbai-embed-large:latest`, which was found installed in local Ollama with
embedding capability and 1024 dimensions. The backend checks the installed
model and derives dimensions from its actual embedding response at startup.
It never downloads a model. The chat model remains the existing `MODEL`
in `config.py`.

- `MAX_HISTORY = 3` in `config.py` limits **every chat request** to the system
  instructions plus the last three conversation messages. Before removing
  older messages, v-6 persists their exact role/content as Redis conversation
  records. Returned history is also trimmed after a completed task.
  There are no LLM history-compaction calls. The latest task and necessary
  recovery/coverage state are attached inside a retained message.
- Each user prompt, model response (including invalid actions), correction,
  tool result and task outcome is stored in Redis.
  Full tool output is saved even when its model-facing preview is truncated.
- Before **every model turn**, Redis searches using the latest user request,
  not a query dominated by old conversation text. It reserves up to three
  candidates for actual shell commands, plus one other tool result and one
  conversation/context candidate (including archived current-task messages).
  Only matches with cosine distance **<= 0.35** are included as hints; this
  initial relevance heuristic may need tuning from real transcripts. If none
  qualifies, no Redis hint text is attached to the LLM request.
  Hints are attached inside a retained message, never added as extra messages,
  and lower-priority hints are dropped if they exceed the context budget.
  Command hints include the original task (for newly
  saved records), arguments, timestamp, outcome and result preview. Duplicate
  actions are collapsed, and linked action records resolve to the latest
  known attempt so an old success is not preferred over a later failure.
  The agent should reuse previously successful testing/execution methods,
  adapting paths and arguments when the task changes; failed hints show
  approaches to correct or avoid. The agent can use `recall(record_id, from_character)` to page
  through a retrieved record without rerunning the original command.
- Before tool execution, an independent exact-action lookup checks prior
  attempts, including failures. Similarity alone never suppresses an action.
  Repeated successful commands and file changes return historical results by
  default, clearly marked as **not executed now**. An identical failed action
  without a retry reason is blocked with its original error and guidance to
  correct arguments or try another method, rather than presenting the failed
  result as completed work.
- A nonempty optional `retry_reason` on `read`, `bash`, `write` or `edit`
  requests a deliberate fresh attempt. Commands and changes still require
  approval. An action declined in the current task cannot be retried in that
  task, even with a reason. Three repeated cache requests without a reason
  stop the task rather than looping.
- File reads validate current byte hashes before reusing exact pages.
  Summary requests restore matching pages across restarts and continue from
  missing coverage. Changed files are read fresh; recalled text by itself is
  never accepted as checked source evidence.
- A pending action record is saved before commands or file changes execute.
  If interrupted before its final result is recorded, subsequent requests
  report an unknown outcome rather than silently repeating the action.
- The terminal displays `Redis semantic search: HIT/MISS` before thinking,
  including retrieved command names, outcomes, timestamps, distances and record
  IDs. `Redis exact-action lookup: HIT/MISS` appears before each tool attempt;
  exact reuse and explicit retry decisions remain separately visible.
  `Context moved to Redis` shows eviction counts, and `Model context` shows
  the exact conversation-message and attached-hint counts sent to Ollama.
- Directory listing requests must attempt a listing of the requested scope
  before finishing. For simple `Get-ChildItem`/`gci`/`ls` commands, a root
  listing cannot satisfy a request for `redis_backend`. Errors can still be
  reported as blockers. Compound listing commands are not scope-parsed;
  use the suggested simple command for the scoped listing step.

Existing v-6 memory is preserved. At startup the backend adds a missing
`tool_name` tag to the existing Search index and backfills its exact-action
records without regenerating their embeddings. Older records did not save the
original task as a separate field; their command/arguments/results still
provide hints. Redis Search indexing may briefly lag these metadata updates.

Memory is isolated by resolved workspace and embedding-model digest/dimensions.
It has no automatic expiry: persistence across Redis/server restarts depends on
your Redis AOF/RDB and volume configuration. Conversation and tool records can
contain sensitive source/output; protect the Redis service and its backups.
Embedding inputs may be truncated to the embedding model's context, but the
full records remain available through bounded recall. Semantic results are
historical hints, not proof of freshness; fresh commands require an explicit
retry. Redis/model failures are surfaced, with no silent in-memory fallback.
Redis is searched each turn, but irrelevant results are not sent to the LLM.
This frees the active conversation window rather than retaining the full
transcript; it adds archival embedding requests and searches, so it does not
necessarily reduce latency. Internal verified source pages still live outside
the model context for evidence checks. A message-count limit is not a token
limit: oversized current messages/state fail explicitly rather than silently
discarding their content.

No tests, Pylance, linting, or live v-6 execution were run for this version;
runtime validation is left to you.

## v-5: chunked-read recovery

The v-4 transcript showed a shell file read timing out while waiting for
approval. A retry returned truncated output, but the model finished by asking
the user to paste the file instead of using the available paginated reader.

v-5 recognizes standalone `Get-Content [-Path|-LiteralPath] <file> [-Raw]`
commands, including quoted paths. After failure or output truncation, it:

- Supplies an explicit `read` cursor, starting at line 1.
- Restricts actions to that exact read cursor or approved writes/edits.
  Premature final answers and repeated raw shell reads are rejected.
- Advances through `next_read`, including column offsets for long lines,
  until EOF. Each page retains the existing line and serialized-byte limits.
- Allows running-summary file updates between pages when requested by the user.
- Releases the recovery restriction after EOF, or after a paginated read itself
  fails so the model can correct or report a genuine blocker.

This is deliberately not a shell parser: pipelines, compound commands and
unrecognized command forms are not automatically converted to file reads.
The original approval timeout behavior is unchanged; recovery avoids requiring
another shell command to inspect the same file.

Coverage also uses a decoded-content hash independently of the original-file
hash. This fixes UTF-16 and UTF-8-BOM files incorrectly appearing incomplete
after all their decoded text has been read. Original-byte hashes still detect
source changes and invalidate cached evidence.

## v-5 validation (prior version)

No Pylance, lint or coverage tooling is required. Focused inline checks exercised
timeout, output truncation and nonzero exit recovery across UTF-8, UTF-8-BOM,
UTF-16 LE and UTF-16 BE files, including CRLF and long-line continuation.
They also checked premature-finish rejection, exact cursor validation, permitted
summary writes, command approval, and missing-file/compound-command handling.

These checks use fake model replies and fake tool failures. A separate live
`qwen2.5-coder:7b` run, after an injected shell timeout, produced two successful
`read` calls covering a 50-line sample, then a final answer identifying its model
and page-size constants. The actual v-4 source also reconstructed exactly from
40 bounded pages in a reader-only check. This is not a claim that arbitrary
prompts or model-generated summaries are always correct.
