


def action_schema(*, bash_description: str = "List the workspace directory only: Get-ChildItem or ls.") -> dict:
    definitions = {
        "read": ("Read exact UTF-8 file text, including newlines.", {"path": "string"}),
        "write": ("Write exact UTF-8 content to a file.", {"path": "string", "content": "string"}),
        "edit": ("Replace exactly one occurrence of old_str.", {
            "path": "string", "old_str": "string", "new_str": "string",
        }),
        "bash": (bash_description, {"command": "string"}),
        "finish": ("Return the final answer when verified complete or blocked.", {"answer": "string"}),
    }
    return {"type": "object", "anyOf": [
        {
            "type": "object", "description": description,
            "properties": {
                "name": {"type": "string", "const": name},
                "arguments": {
                    "type": "object",
                    "properties": {key: {"type": kind} for key, kind in arguments.items()},
                    "required": list(arguments), "additionalProperties": False,
                },
            },
            "required": ["name", "arguments"], "additionalProperties": False,
        }
        for name, (description, arguments) in definitions.items()
    ]}