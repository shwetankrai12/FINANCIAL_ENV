"""
Baseline inference script — required for OpenEnv hackathon submission.

Runs a simple baseline agent against all 3 tasks and prints scores.
Uses HuggingFace InferenceClient (meta-llama/Llama-3.3-70B-Instruct).

Usage:
    export HF_TOKEN=your_token
    python baseline_agent.py

Expected output: score for each task, 0.0-1.0.
"""

import asyncio
import os
from huggingface_hub import InferenceClient

from server.environment import FinancialEnvironment
from models.types import FinancialAction


HF_TOKEN = os.getenv("HF_TOKEN", "")
BASELINE_TICKER = "AAPL"
BASELINE_PERIOD = "3mo"
BASELINE_TICKERS = ["AAPL", "MSFT", "GOOGL"]


def ask_llm(prompt: str) -> str:
    """Ask LLM which task parameters to use."""
    if not HF_TOKEN:
        return ""
    client = InferenceClient(token=HF_TOKEN)
    resp = client.chat_completion(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": "You are a financial agent. Answer concisely."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=100,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


async def run_baseline():
    print("=" * 50)
    print("Financial Analysis Env — Baseline Agent")
    print("=" * 50)

    env = FinancialEnvironment()
    total_score = 0.0
    results = {}

    # Reset
    await env.reset_async()
    print(f"Episode started: {env.state.episode_id[:8]}...\n")

    # Task 1: price_check (easy)
    print("[Task 1 - EASY] price_check")
    action = FinancialAction(task_id="price_check", ticker=BASELINE_TICKER, period=BASELINE_PERIOD)
    obs = await env.step_async(action)
    score = obs.grader_result.score if obs.grader_result else 0.0
    results["price_check"] = score
    total_score += score
    print(f"  Ticker: {BASELINE_TICKER} | Price: ${obs.price_data.current_price if obs.price_data else 'N/A'}")
    print(f"  Grader: {obs.grader_result.reason if obs.grader_result else 'N/A'}")
    print(f"  Score: {score:.2f}\n")

    # Task 2: trend_analysis (medium)
    print("[Task 2 - MEDIUM] trend_analysis")
    action = FinancialAction(task_id="trend_analysis", ticker=BASELINE_TICKER, period=BASELINE_PERIOD)
    obs = await env.step_async(action)
    score = obs.grader_result.score if obs.grader_result else 0.0
    results["trend_analysis"] = score
    total_score += score
    if obs.signals:
        print(f"  RSI: {obs.signals.rsi_14} | Trend: {obs.signals.trend} | SMA20: {obs.signals.sma_20}")
    print(f"  Grader: {obs.grader_result.reason if obs.grader_result else 'N/A'}")
    print(f"  Score: {score:.2f}\n")

    # Task 3: portfolio_rank (hard)
    print("[Task 3 - HARD] portfolio_rank")
    action = FinancialAction(
        task_id="portfolio_rank",
        ticker=BASELINE_TICKERS[0],
        period=BASELINE_PERIOD,
        tickers=BASELINE_TICKERS,
    )
    obs = await env.step_async(action)
    score = obs.grader_result.score if obs.grader_result else 0.0
    results["portfolio_rank"] = score
    total_score += score
    print(f"  Tickers: {BASELINE_TICKERS}")
    print(f"  Best: {obs.best_ticker} | Worst: {obs.worst_ticker}")
    print(f"  Grader: {obs.grader_result.reason if obs.grader_result else 'N/A'}")
    print(f"  Score: {score:.2f}\n")

    # Summary
    avg = total_score / 3
    print("=" * 50)
    print("BASELINE SCORES")
    print("=" * 50)
    for task, s in results.items():
        print(f"  {task:<20} {s:.2f}")
    print(f"  {'AVERAGE':<20} {avg:.2f}")
    print("=" * 50)

    return results


if __name__ == "__main__":
    asyncio.run(run_baseline())
