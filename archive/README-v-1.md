# Local demo v-1

The original `demo.py` and its helpers are unchanged. `v-1.py` reuses its Rich
presentation, approvals and command/file executor, but replaces the agent loop
and read protocol.

```powershell
python .\v-1.py --workspace .
```

Enter `read demo.py file and summarise its content`. Chat continues until `/bye`.

- Default context: 32,768 tokens, with 512 output tokens and a 2,048-token
  template reserve. Request timeout: 900 seconds; task timeout: 2,400 seconds;
  maximum actions: 64. Override using `--context`, `--request-timeout`,
  `--timeout` and `--max-requests`.
- `read` takes `path`, `from_line`, `to_line` (1-based, inclusive). Pages cap
  at 40 lines and 2,000 serialized bytes. Responses include numbered text,
  total lines, a file hash, and `next_line`/`next_column` for continuation.
  Ordinary lines are kept whole; oversized individual lines are continued
  with optional 0-based `from_column`. Follow the exact `next_read` arguments;
  a first-missing-range ledger also identifies gaps. Text is not silently
  discarded. Labels are metadata, not text to copy into files.
- Prompt budgeting uses a conservative serialized-byte bound, not a guessed
  characters-per-token ratio. Older exchanges are summarized by the model
  before eviction using bounded plain-text notes. Compaction is visible and
  truncated notes are explicitly flagged as incomplete; failures are explicit.
  Context limits are checked before each request.
- Repeated unchanged read starts are blocked and receive continuation feedback
  without appending the same source again. Three consecutive repeats stop with
  an explicit error instead of an endless loop.
- File-based final answers must provide exact quotes from observed source
  lines. Source fragments are also reconstructed and compared against the
  file's SHA-256 hash to display complete versus partial read coverage.
  Quote validation checks provenance, not the truth of every claim.
  Invalid quotes receive exact checked alternatives without display labels.
  The finish schema itself restricts quote values to checked observed-source
  choices, rather than relying only on an instruction not to invent citations.
  Observed pages stay available across follow-up prompts; changed files invalidate
  old evidence. Review the answer against the source.
- Commands and changes require approval and are not sandboxed. Command output
  previews are explicitly marked partial when too large for the tool budget.

This still uses JSON-schema actions, not native tool-call parsing. Larger
context windows can be slower and require more memory. No dependencies beyond
the existing demo's Rich dependency are added.

## Verification on the current local model

The exact task `read demo.py file and summarise its content` was exercised with
`qwen2.5-coder:7b`. Thirteen paginated reads covered all 311 lines, verified by
reconstructing the source and matching its SHA-256 hash. An earlier unconstrained
finish attempt invented citations and was rejected. With source-constrained
evidence and the actual read state retained, a correction follow-up produced a
valid summary describing the interactive tool loop, approval handling,
file/command actions, CLI arguments, error panels, and main entry point.
Its source quotes were checked against the actual file, and the summary was
reviewed against the implementation.

This is a verified full-read plus correction workflow, **not a demonstrated
fresh one-pass guarantee**. The model ran on CPU (`size_vram=0`): the full-read
attempt took about 30 minutes including citation rejection; the final corrected
summary took about 10 minutes including context compaction. The larger timeout
defaults accommodate this latency, but do not make inference faster.

Offline checks also exercised page byte/line thresholds, CRLF and long-line
reconstruction, empty files, range errors, repeated-read correction, history
compaction bounds, truncated-note warnings, stale evidence, source-cache
invalidation, command timeouts, Unicode previews and `/bye` continuation.
No test fixtures, test files, linting or coverage tooling were added.
