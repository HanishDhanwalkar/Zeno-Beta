# Local demo v-2

Run from `local_tests`:

```powershell
python .\v-2.py --workspace .
```

The executable code in `v-1.py` is unchanged; only its docstring records the
failure table. Version 2 keeps the existing bounded reads, context budgets,
source evidence, Rich panels, approvals and continuous chat until `/bye`.

Changes:

- General chat instructions explicitly permit `finish` without any tool calls.
  Workspace and model identity are supplied as known runtime facts.
  File-summary instructions apply only when source inspection is requested.
- File tools reject directories, including the workspace root, with a precise
  explanation. Paths outside the selected workspace remain rejected.
  Writes/edits are checked before reaching the original approved executor.
- Identical failed actions are tracked for reads, writes, edits and commands.
  Only the first attempt executes. Later attempts get the original failure and
  corrective feedback; the third identical attempt stops the task explicitly.
  Equivalent absolute/relative file paths and omitted zero column offsets are
  normalized. Different arguments may be used for recovery. A new user prompt
  resets failed-action tracking, allowing intentional retries.

This is not a guarantee that the local model will always select an appropriate
action. A valid schema ensures action shape, not user-intent alignment.
Commands and file changes require approval; commands are not sandboxed.
Context and timeout settings remain configurable as in v-1.

## Verification

Live verification with the unchanged `qwen2.5-coder:7b`:

| User prompt | Result | Tool calls | Model requests |
|---|---|---|---|
| what is current workspace | Correct workspace path | 0 | 1 |
| what model are you using | qwen2.5-coder:7b | 0 | 1 |

Both prompts ran in the same chat, which then exited on `/bye`.
Offline checks verified directory rejection before writes, absolute/relative
failed-read normalization, a three-attempt stop with only one execution,
failed-command blocking, recovery with corrected file arguments, source quotes,
and outside-workspace rejection. An AST fingerprint confirmed that executable
code in `v-1.py` did not change.
