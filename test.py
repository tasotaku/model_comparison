"""
minimal_ab_test.py – Compare two OpenAI Chat models asynchronously, record latency, token usage, and cost.

Usage::
    python minimal_ab_test.py -p "こんにちは、自己紹介してください" -m1 gpt-4o -m2 gpt-4.1-nano

The script appends a JSON‑line to logs/run.jsonl per call, containing:
    timestamp, model, prompt, response, latency_ms, prompt_tokens, completion_tokens, total_cost_usd

Adjust PRICING table as prices change.
"""
import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from openai import AsyncOpenAI

# ──────────────────────────────────────────────────────────────────────────────
# 1. Pricing table (USD). Values are **per‑million tokens** to keep precision.
#    Ref: https://openai.com/api/pricing – GPT‑4o $5 / 1M input, $20 / 1M output
#    Update as needed.
# ──────────────────────────────────────────────────────────────────────────────
PRICING_PER_MILLION = {
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}

def calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return USD cost for a call."""
    if model not in PRICING_PER_MILLION:
        raise KeyError(f"Unknown model {model}. Add it to PRICING_PER_MILLION.")
    price = PRICING_PER_MILLION[model]
    
    # Debug information
    print(f"\nCost calculation for {model}:")
    print(f"Prompt tokens: {prompt_tokens} @ ${price['input']}/1M tokens")
    print(f"Completion tokens: {completion_tokens} @ ${price['output']}/1M tokens")
    
    prompt_cost = (prompt_tokens * price["input"]) / 1_000_000
    completion_cost = (completion_tokens * price["output"]) / 1_000_000
    total_cost = prompt_cost + completion_cost
    
    print(f"Prompt cost: ${prompt_cost:.6f}")
    print(f"Completion cost: ${completion_cost:.6f}")
    print(f"Total cost: ${total_cost:.6f}")
    
    return total_cost


async def ask_one_async(prompt: str, model: str, client: AsyncOpenAI) -> dict:
    """Send a prompt asynchronously, measure latency, and collect metadata."""
    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = (time.perf_counter() - start) * 1000

    usage = response.usage
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


def print_comparison(results: List[dict]) -> None:
    """Print a side-by-side comparison of model results."""
    print("\n===== Model Comparison =====")
    print("\n----- Responses -----")
    for result in results:
        print(f"\n[{result['model']}]")
        print(result["response"])
    
    print("\n----- Metrics Comparison -----")
    print(f"{'Metric':<15} {'Model 1':<15} {'Model 2':<15}")
    print("-" * 45)
    
    metrics = [
        ("Latency (ms)", "latency_ms"),
        ("Prompt tokens", "prompt_tokens"),
        ("Completion tok.", "completion_tokens"),
        ("Total tokens", "total_tokens"),
        ("Cost (USD)", "cost_usd"),
    ]
    
    for label, key in metrics:
        values = [f"{result[key]:.2f}" for result in results]
        print(f"{label:<15} {values[0]:<15} {values[1]:<15}")


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Compare two OpenAI models asynchronously.")
    parser.add_argument("-p", "--prompt", required=True, help="User prompt text")
    parser.add_argument("-m1", "--model1", default="gpt-4o", help="First model name")
    parser.add_argument("-m2", "--model2", default="gpt-4.1-nano", help="Second model name")
    parser.add_argument("-l", "--logfile", default="logs/run.jsonl", help="Path to JSONL log file")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY env var not set")

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Run both models concurrently
    tasks = [
        ask_one_async(args.prompt, args.model1, client),
        ask_one_async(args.prompt, args.model2, client)
    ]
    results = await asyncio.gather(*tasks)

    # Log results
    log_path = Path(args.logfile)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Print comparison
    print_comparison(results)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
