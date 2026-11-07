# v-3: reliable bash tool calling

Run: `python .\v-3.py --workspace .` (type `/bye` to exit). Commands still need approval.

## Changes from v-2
- `bash` schema description with PowerShell/Bash examples (count lines, list files).
- System prompt "BASH RULES": run/execute requests must call `bash`; "that command" is resolved from the previous answer; listing/counting uses `bash`, never `read`.
- Harness detects run intent (negations such as "don't run it" are ignored), passes the previous assistant answer as data, and rejects a `finish` that skips `bash` (up to 2 corrections).
- Directory error no longer tells the model to avoid tools.
- v-2 guards (failed-action blocking, direct answers for known facts) are retained.

## Verification
- Offline (fake model): `finish` with only a command is corrected into `bash`, then a final answer.
- Live `qwen2.5-coder:7b`, two turns: "how can I count total lines..." then "you can run that command with bash tool" -> `bash` called with `(Get-ChildItem -Recurse -File | Get-Content | Measure-Object -Line).Lines`, result fed back, final answer given.
- Limitation: small model, so the command it proposes is not guaranteed correct; approval stays in the loop.
