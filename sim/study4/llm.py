"""Minimal Ollama chat client with schema-constrained JSON output. Stdlib only."""
from __future__ import annotations

import json
import statistics
import urllib.request
from collections import Counter
from dataclasses import dataclass, field

URL = "http://127.0.0.1:11434/api/chat"


@dataclass
class Capability:
    """Everything here is capability. Nothing here grants authority."""
    model: str
    num_ctx: int = 8192
    effort: int = 1            # samples per decision (majority / median)

    @property
    def tag(self) -> str:
        return f"{self.model}|ctx{self.num_ctx}|n{self.effort}"


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    parse_failures: int = 0
    transcript: list = field(default_factory=list)


def _once(cap: Capability, system: str, user: str, schema: dict, seed: int, temperature: float,
          usage: Usage, max_tokens: int) -> dict | None:
    body = {"model": cap.model, "stream": False, "format": schema, "keep_alive": "30m",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": temperature, "seed": seed, "num_ctx": cap.num_ctx,
                        "num_predict": max_tokens}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    usage.calls += 1
    usage.prompt_tokens += out.get("prompt_eval_count", 0)
    usage.output_tokens += out.get("eval_count", 0)
    usage.seconds += out.get("total_duration", 0) / 1e9
    text = out["message"]["content"]
    usage.transcript.append({"user": user, "reply": text, "seed": seed})
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        usage.parse_failures += 1
        return None


def ask(cap: Capability, system: str, user: str, schema: dict, seed: int, usage: Usage,
        max_tokens: int = 160) -> dict | None:
    """
    One decision. effort 1 is greedy decoding. effort n samples n replies and
    aggregates field by field: median for numbers, majority for the rest.
    """
    if cap.effort == 1:
        return _once(cap, system, user, schema, seed, 0.0, usage, max_tokens)
    replies = [r for i in range(cap.effort)
               if (r := _once(cap, system, user, schema, seed * 1000 + i, 0.8, usage, max_tokens))]
    if not replies:
        return None
    agg = {}
    for key in schema["properties"]:
        vals = [r[key] for r in replies if key in r]
        if not vals:
            continue
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            agg[key] = statistics.median_low(vals)
        else:
            agg[key] = Counter(json.dumps(v) for v in vals).most_common(1)[0][0]
            agg[key] = json.loads(agg[key])
    return agg


def available_models() -> list[str]:
    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=10) as r:
        return [m["name"] for m in json.loads(r.read())["models"]]
