import json
from pathlib import Path

from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from config import MODEL, OLLAMA_URL

LOCAL_HTTP = build_opener(ProxyHandler({}))


def workspace_path(cwd: Path, path: str) -> Path:
    target = (cwd / path).resolve()
    if not target.is_relative_to(cwd.resolve()) or target == cwd:
        raise ValueError("File path must stay inside the temporary workspace.")
    return target


def request_json(url: str, *, payload: dict | None = None, timeout: float = 10) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        with LOCAL_HTTP.open(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Cannot reach local Ollama at {OLLAMA_URL}: {exc.reason}. "
            "Start the Ollama application or run 'ollama serve', then retry."
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(f"Ollama request timed out after {timeout:g}s.") from exc
    if not isinstance(result, dict):
        raise ValueError("Ollama returned a non-object JSON response.")
    if result.get("error"):
        raise RuntimeError(f"Ollama error: {result['error']}")
    return result


def check_ollama() -> None:
    tags = request_json(f"{OLLAMA_URL}/api/tags")
    installed = {entry["name"] for entry in tags["models"]}
    if MODEL not in installed:
        raise RuntimeError(
            f"Model {MODEL!r} is not installed. Run 'ollama pull {MODEL}' "
            "if you want to install it; this script does not download models."
        )
    print(f"OLLAMA READY | {OLLAMA_URL} | {MODEL}", flush=True)
