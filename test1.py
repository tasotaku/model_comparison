"""
minimal_ab_test.py – Call one OpenAI Chat model, record latency, token usage, and cost.

Usage::
    python minimal_ab_test.py -p "こんにちは、自己紹介してください" -m gpt-4o

The script appends a JSON‑line to logs/run.jsonl per call, containing:
    timestamp, model, prompt, response, latency_ms, prompt_tokens, completion_tokens, total_cost_usd

Adjust PRICING table as prices change.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# 1. Pricing table (USD). Values are **per‑million tokens** to keep precision.
#    Ref: https://openai.com/api/pricing – GPT‑4o $5 / 1M input, $20 / 1M output
#    Update as needed.
# ──────────────────────────────────────────────────────────────────────────────
PRICING_PER_MILLION = {
    "gpt-4o": {"input": 5.0, "output": 20.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}


def calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return USD cost for a call."""
    if model not in PRICING_PER_MILLION:
        raise KeyError(f"Unknown model {model}. Add it to PRICING_PER_MILLION.")
    price = PRICING_PER_MILLION[model]
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000


def ask_one(prompt: str, model: str = "gpt-4o") -> dict:
    """Send a prompt, measure latency, and collect metadata."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = (time.perf_counter() - start) * 1000

    usage = response.usage  # prompt_tokens, completion_tokens, total_tokens
    cost_usd = calc_cost(model, usage.prompt_tokens, usage.completion_tokens)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "prompt": prompt,
        "response": response.choices[0].message.content,
        "latency_ms": round(latency_ms, 2),
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "cost_usd": round(cost_usd, 6),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Call an OpenAI model once and log metrics.")
    parser.add_argument("-p", "--prompt", required=True, help="User prompt text")
    parser.add_argument("-m", "--model", default="gpt-4o", help="Model name")
    parser.add_argument("-l", "--logfile", default="logs/run.jsonl", help="Path to JSONL log file")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY env var not set")

    record = ask_one(args.prompt, args.model)

    log_path = Path(args.logfile)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Quick stdout summary
    print("\n===== Chat Completion =====")
    print(record["response"])
    print("\n----- Metrics -----")
    print(f"Model          : {record['model']}")
    print(f"Latency (ms)   : {record['latency_ms']}")
    print(f"Prompt tokens  : {record['prompt_tokens']}")
    print(f"Completion tok.: {record['completion_tokens']}")
    print(f"Cost (USD)     : ${record['cost_usd']}")


if __name__ == "__main__":
    main()
