"""v-7 memory: one incrementally saved user/assistant exchange per query.

Reuses v-1 connection, embeddings and exact-action helpers. Paired exchanges
have a separate Search index; step records are not semantic search results.
Existing v-6 records are preserved, not automatically converted into pairs.
"""

from __future__ import annotations

import json
import runpy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import redis

BaseMemory = runpy.run_path(str(Path(__file__).with_name("v-1.py")))["RedisMemory"]


class RedisMemory(BaseMemory):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prefix = self.prefix.replace("zeno:v6:", "zeno:v7:", 1)
        self.index = self.prefix + "index"
        try:
            try:
                self.client.execute_command("FT.INFO", self.index)
            except redis.ResponseError as exc:
                if not any(text in str(exc).lower() for text in (
                    "unknown index name", "no such index", "search_index_not_found",
                )):
                    raise
                self.client.execute_command(
                    "FT.CREATE", self.index, "ON", "HASH", "PREFIX", 1,
                    self.prefix + "record:", "SCHEMA",
                    "kind", "TAG", "turn_id", "TAG",
                    "vector", "VECTOR", "HNSW", 6,
                    "TYPE", "FLOAT32", "DIM", self.dimension,
                    "DISTANCE_METRIC", "COSINE",
                )
        except redis.RedisError as exc:
            self.client.close()
            raise RuntimeError(f"Paired Redis memory startup failed: {exc}") from exc

    def add(
        self, kind: str, data: dict, *, turn_id: str, prompt: str,
        action_key: str | None = None, timeout: float | None = None,
    ) -> dict:
        key = self.prefix + "record:" + turn_id
        body = self.client.hget(key, "body")
        if body is None:
            exchange = {
                "id": turn_id, "at": datetime.now(timezone.utc).isoformat(),
                "turn_id": turn_id, "kind": "exchange", "prompt": prompt,
                "data": {
                    "user": prompt, "assistant": None, "status": "pending",
                    "error": None, "tool_steps": [], "events": [],
                },
            }
        else:
            exchange = json.loads(body)
        pair = exchange["data"]
        event = {
            "id": uuid4().hex, "at": datetime.now(timezone.utc).isoformat(),
            "turn_id": turn_id, "kind": kind, "prompt": prompt,
            "action_key": action_key, "data": data,
        }
        if kind in {"tool", "action"}:
            pair["tool_steps"].append(event)
        else:
            pair["events"].append(event)
        if kind == "task":
            pair["assistant"] = data["answer"]
            pair["error"] = data["error"]
            pair["status"] = "failed" if data["error"] else "completed"
        exchange["updated_at"] = event["at"]
        methods = [
            {
                "name": step["data"]["name"],
                "arguments": step["data"]["arguments"],
                "outcome": step["data"]["outcome"],
            }
            for step in pair["tool_steps"]
        ]
        vector = self._vector_bytes(
            "User: " + prompt[:900]
            + "\nAssistant: " + (pair["assistant"] or "")[:900]
            + "\nMethods: " + json.dumps(methods, ensure_ascii=True)[:1400],
            timeout=timeout,
        )
        with self.client.pipeline(transaction=True) as pipe:
            pipe.hset(key, mapping={
                "body": json.dumps(exchange, ensure_ascii=True), "vector": vector,
                "turn_id": turn_id, "kind": "exchange",
            })
            if action_key is not None:
                step_key = self.prefix + "step:" + event["id"]
                pipe.hset(step_key, mapping={"body": json.dumps(event, ensure_ascii=True)})
                pipe.hset(self.prefix + "actions", action_key, step_key)
            pipe.execute()
        return exchange

    def retrieve(
        self, query: str, *, turn_id: str, limit: int = 3,
        timeout: float | None = None, max_distance: float = 0.35,
    ) -> list[dict]:
        vector = self._vector_bytes(
            "Represent this sentence for searching relevant passages: " + query[:1800],
            timeout=timeout,
        )
        results = self._search(
            vector, f"@kind:{{exchange}} -@turn_id:{{{turn_id}}}", limit=limit,
        )
        hints = []
        for record, distance in results:
            if distance > max_distance:
                continue
            pair = record["data"]
            user = pair["user"]
            assistant = pair["assistant"]
            steps = pair["tool_steps"]
            previews = []
            for step in steps[-3:]:
                entry = step["data"]
                previews.append({
                    "kind": step["kind"],
                    "name": entry["name"], "outcome": entry["outcome"],
                    "arguments_preview": json.dumps(entry["arguments"], ensure_ascii=True)[:500],
                    "result_preview": json.dumps(entry["result"], ensure_ascii=True)[:500],
                    "reused": entry.get("reused", False),
                })
            hints.append({
                "id": record["id"], "at": record["at"], "kind": "exchange",
                "distance": distance, "user": user[:700],
                "assistant": assistant[:700] if assistant is not None else None,
                "status": pair["status"],
                "error": pair["error"][:500] if pair["error"] is not None else None,
                "tool_steps": previews,
                "preview_is_partial": (
                    len(user) > 700 or len(assistant or "") > 700
                    or len(pair["error"] or "") > 500 or bool(steps)
                ),
                "guidance": "User and assistant are one historical exchange. "
                "Pending exchanges have no final answer. Tool previews are partial; "
                "recall the exchange for details. Match current scope and check exact "
                "actions before retrying; historical success is not fresh execution.",
            })
        return hints
