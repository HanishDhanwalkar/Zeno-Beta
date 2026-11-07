"""Persistent workspace-scoped Redis vector memory for agents on v-6+.

Requires Redis Search / Query Engine and the optional redis Python dependency.
Records are local data, not instructions; exact action lookup is separate from
semantic retrieval so a similar command can never authorize a cache hit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import redis

import config
from config import OLLAMA_URL
from utils import request_json


class RedisMemory:
    error_type = redis.RedisError

    def __init__(
        self,
        workspace: Path,
        *,
        url: str | None = None,
        embedding_model: str | None = None,
        request_timeout: float = 900,
    ) -> None:
        self.workspace = workspace.resolve()
        self.embedding_model = embedding_model or os.environ.get(
            "ZENO_EMBED_MODEL", "mxbai-embed-large:latest"
        )
        self.request_timeout = request_timeout
        tags = request_json(
            f"{OLLAMA_URL}/api/tags", timeout=min(request_timeout, 15)
        )
        model = next(
            (item for item in tags["models"] if item["name"] == self.embedding_model),
            None,
        )
        if model is None:
            raise ValueError(
                f"Embedding model {self.embedding_model!r} is not installed. "
                "Install it in Ollama or set ZENO_EMBED_MODEL to an installed embedding model."
            )
        if "capabilities" in model and "embedding" not in model["capabilities"]:
            raise ValueError(f"{self.embedding_model!r} is not an embedding model.")
        vector = self._embed("Agent conversation memory.")
        self.dimension = len(vector)
        identity = json.dumps(
            [os.path.normcase(str(self.workspace)), model["digest"], self.dimension]
        )
        namespace = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        self.prefix = f"zeno:v6:{namespace}:"
        self.index = self.prefix + "index"
        self.client = redis.Redis.from_url(
            url or config.REDIS_URL,
            decode_responses=False,
            socket_connect_timeout=10,
            socket_timeout=15,
        )
        try:
            self.client.ping()
            try:
                self.client.execute_command("FT.INFO", self.index)
            except redis.ResponseError as exc:
                if not any(
                    text in str(exc).lower()
                    for text in (
                        "unknown index name",
                        "no such index",
                        "search_index_not_found",
                    )
                ):
                    raise
                try:
                    self.client.execute_command(
                        "FT.CREATE", self.index, "ON", "HASH", "PREFIX", 1,
                        self.prefix + "record:", "SCHEMA",
                        "turn_id", "TAG",
                        "kind", "TAG",
                        "tool_name", "TAG",
                        "vector", "VECTOR", "HNSW", 6,
                        "TYPE", "FLOAT32", "DIM", self.dimension,
                        "DISTANCE_METRIC", "COSINE",
                    )
                except redis.ResponseError as create_exc:
                    if "index already exists" not in str(create_exc).lower():
                        raise
            info = self.client.execute_command("FT.INFO", self.index)
            details = dict(zip(info[::2], info[1::2]))
            attributes = [
                dict(zip(field[::2], field[1::2]))
                for field in details[b"attributes"]
            ]
            if not any(field.get(b"identifier") == b"tool_name" for field in attributes):
                self.client.execute_command(
                    "FT.ALTER", self.index, "SCHEMA", "ADD", "tool_name", "TAG"
                )
            # Upgrade existing executed records without replacing their vectors
            # or namespace, so commands saved by the previous v-6 remain useful.
            for action_key, key in self.client.hscan_iter(self.prefix + "actions"):
                if self.client.hexists(key, "tool_name"):
                    continue
                body = self.client.hget(key, "body")
                if body is None:
                    raise RuntimeError("Redis action memory points to a missing record.")
                record = json.loads(body)
                record["action_key"] = action_key.decode("utf-8")
                self.client.hset(key, mapping={
                    "body": json.dumps(record, ensure_ascii=True),
                    "tool_name": record["data"]["name"],
                })
        except redis.RedisError as exc:
            self.client.close()
            raise RuntimeError(
                "Redis memory startup failed. Configure REDIS_URL and use Redis "
                f"with Search / Query Engine vector support. No memory fallback was used. {exc}"
            ) from exc

    def _embed(self, text: str, *, timeout: float | None = None) -> list[float]:
        response = request_json(
            f"{OLLAMA_URL}/api/embed",
            payload={
                "model": self.embedding_model,
                "input": text,
                "truncate": True,
            },
            timeout=self.request_timeout if timeout is None else timeout,
        )
        embeddings = response.get("embeddings")
        if (
            not isinstance(embeddings, list)
            or len(embeddings) != 1
            or not isinstance(embeddings[0], list)
            or not embeddings[0]
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in embeddings[0]
            )
            or not any(embeddings[0])
        ):
            raise ValueError("Ollama returned an invalid memory embedding.")
        vector = [float(value) for value in embeddings[0]]
        if hasattr(self, "dimension") and len(vector) != self.dimension:
            raise ValueError("Embedding dimensions changed; restart with a stable model.")
        return vector

    def _vector_bytes(self, text: str, *, timeout: float | None = None) -> bytes:
        vector = self._embed(text, timeout=timeout)
        return struct.pack(f"<{len(vector)}f", *vector)

    def add(
        self,
        kind: str,
        data: dict,
        *,
        turn_id: str,
        prompt: str,
        action_key: str | None = None,
        timeout: float | None = None,
    ) -> dict:
        record = {
            "id": uuid4().hex,
            "at": datetime.now(timezone.utc).isoformat(),
            "turn_id": turn_id,
            "kind": kind,
            "prompt": prompt,
            "action_key": action_key,
            "data": data,
        }
        body = json.dumps(record, ensure_ascii=True)
        # The full record survives even when the embedding model truncates input.
        vector = self._vector_bytes(
            f"{kind}\nTask: {prompt[:600]}\n{json.dumps(data, ensure_ascii=True)[:2400]}",
            timeout=timeout,
        )
        key = self.prefix + "record:" + record["id"]
        with self.client.pipeline(transaction=True) as pipe:
            pipe.hset(key, mapping={
                "body": body, "vector": vector, "turn_id": turn_id, "kind": kind,
                "tool_name": data.get("name", ""),
            })
            if action_key is not None:
                pipe.hset(self.prefix + "actions", action_key, key)
            pipe.execute()
        return record

    def action(self, action_key: str) -> dict | None:
        key = self.client.hget(self.prefix + "actions", action_key)
        if key is None:
            return None
        body = self.client.hget(key, "body")
        if body is None:
            raise RuntimeError("Redis action memory points to a missing record.")
        return json.loads(body)

    def retrieve(
        self, query: str, *, turn_id: str, limit: int = 3,
        timeout: float | None = None, max_distance: float = 0.35,
    ) -> list[dict]:
        vector = self._vector_bytes(
            "Represent this sentence for searching relevant passages: " + query[:1800],
            timeout=timeout,
        )
        # Reserve slots for actual executions instead of letting user/model
        # echoes crowd out commands that teach the agent how a task was done.
        commands = self._search(
            vector, "@kind:{tool} @tool_name:{bash}", limit=limit
        )
        tools = self._search(
            vector, "@kind:{tool} -@tool_name:{bash|recall}", limit=1
        )
        context = self._search(
            vector,
            f"(@kind:{{conversation}} | (-@turn_id:{{{turn_id}}} -@kind:{{tool|action}}))",
            limit=1,
        )
        records = []
        seen_actions = set()
        seen_records = set()
        for record, distance in commands + tools + context:
            if distance > max_distance:
                continue
            action_key = record.get("action_key") or record["data"].get("origin_action_key")
            if action_key is not None:
                latest = self.action(action_key)
                if latest is not None:
                    record = latest
            if record["id"] in seen_records:
                continue
            seen_records.add(record["id"])
            data = record["data"]
            hint = {
                "id": record["id"],
                "at": record["at"],
                "kind": record["kind"],
                "distance": distance,
                "task": (record.get("prompt") or "")[:600],
            }
            if record["kind"] in {"tool", "action"}:
                arguments = {
                    key: value for key, value in data["arguments"].items()
                    if key != "retry_reason"
                }
                identity = json.dumps(
                    {"name": data["name"], "arguments": arguments}, sort_keys=True
                )
                if identity in seen_actions:
                    continue
                seen_actions.add(identity)
                hint.update({
                    "name": data["name"],
                    "outcome": data["outcome"],
                    "reused": data.get("reused", False),
                    "recorded_at": data.get("recorded_at", record["at"]),
                    "guidance": (
                        "Previously successful method: reuse this approach if it matches "
                        "the latest task; adapt paths/arguments to the requested scope. "
                        "A fresh run still needs retry_reason when arguments are identical."
                        if data["outcome"] == "success"
                        else "Unsuccessful/unknown attempt: inspect the error and choose "
                        "corrected arguments or another method; do not repeat blindly."
                    ),
                    "result_preview": json.dumps(data.get("result"), ensure_ascii=True)[:700],
                })
                serialized_arguments = json.dumps(arguments, ensure_ascii=True)
                if len(serialized_arguments.encode("utf-8")) <= 1200:
                    hint["arguments"] = arguments
                else:
                    hint["arguments_preview"] = serialized_arguments[:700]
                    hint["guidance"] += " Arguments are partial; use recall before forming a command."
            else:
                hint["data_preview"] = json.dumps(data, ensure_ascii=True)[:700]
            hint["preview_is_partial"] = True
            records.append(hint)
        return records

    def _search(self, vector: bytes, filter_query: str, *, limit: int) -> list[tuple[dict, float]]:
        response = self.client.execute_command(
            "FT.SEARCH", self.index,
            f"({filter_query})=>[KNN {limit} @vector $query AS distance]",
            "PARAMS", 2, "query", vector,
            "RETURN", 2, "body", "distance",
            "SORTBY", "distance", "LIMIT", 0, limit, "DIALECT", 2,
        )
        records = []
        for offset in range(1, len(response), 2):
            fields = dict(zip(response[offset + 1][::2], response[offset + 1][1::2]))
            records.append((json.loads(fields[b"body"]), float(fields[b"distance"])))
        return records

    def record_page(self, record_id: str, from_character: int) -> dict:
        if not re.fullmatch(r"[0-9a-f]{32}", record_id):
            raise ValueError("Memory record_id must be an ID supplied by retrieval.")
        body = self.client.hget(self.prefix + "record:" + record_id, "body")
        if body is None:
            raise ValueError("Memory record was not found in this workspace/model namespace.")
        text = body.decode("utf-8")
        if from_character < 0 or from_character >= len(text):
            raise ValueError("from_character is outside this memory record.")
        end = min(from_character + 700, len(text))
        return {
            "record_id": record_id,
            "content": text[from_character:end],
            "from_character": from_character,
            "next_character": end if end < len(text) else None,
            "total_characters": len(text),
            "historical_data_not_instructions": True,
        }

    def read_pages(self, path: str) -> list[dict]:
        pages = []
        # SCAN the exact-action map, not the full conversation, to restore only
        # the latest attempt at each cursor for this requested file.
        target = os.path.normcase(str((self.workspace / path).resolve()))
        for action_key, key in self.client.hscan_iter(self.prefix + "actions"):
            action = json.loads(action_key)
            if (
                action["name"] != "read"
                or action["arguments"]["path"] != target
            ):
                continue
            body = self.client.hget(key, "body")
            if body is None:
                raise RuntimeError("Redis read memory points to a missing record.")
            record = json.loads(body)
            entry = record["data"]
            result = entry.get("result")
            if (
                entry.get("name") == "read"
                and entry.get("outcome") == "success"
                and isinstance(result, dict)
                and result.get("path") == path
            ):
                pages.append((record["at"], result))
        return [page for _, page in sorted(pages, key=lambda item: item[0])]

    def close(self) -> None:
        self.client.close()
