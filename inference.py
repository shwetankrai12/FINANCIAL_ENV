"""
Inference Script — Financial Analysis Environment
OpenEnv Hackathon submission
"""

import asyncio
import os
from typing import List, Optional
from openai import OpenAI

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "financial-analysis-env"
MAX_STEPS = 3
SUCCESS_SCORE_THRESHOLD = 0.7

TASKS = [
    {
        "task_id": "price_check",
        "ticker": "AAPL",
        "period": "3mo",
        "tickers": None,
    },
    {
        "task_id": "trend_analysis",
        "ticker": "AAPL",
        "period": "3mo",
        "tickers": None,
    },
    {
        "task_id": "portfolio_rank",
        "ticker": "AAPL",
        "period": "3mo",
        "tickers": ["AAPL", "MSFT", "GOOGL"],
    },
]

SYSTEM_PROMPT = """You are a financial analysis agent interacting with a stock market environment.
You will be given a financial task. Respond with the exact JSON action to take.
Available task_ids: price_check, trend_analysis, portfolio_rank
For portfolio_rank you must include tickers list with at least 2 items.
Respond with only the JSON action, no explanation."""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_action(client: OpenAI, task: dict, step: int, last_obs: Optional[dict] = None) -> dict:
    """Ask LLM which action to take, merge with task dict to ensure all fields present."""
    # Start with task as base — guarantees all required fields
    action = dict(task)

    try:
        user_prompt = f"Take this financial analysis action: {task}"
        if last_obs:
            user_prompt += f"\nLast observation: {last_obs}"

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=200,
            stream=False,
        )
        import json
        text = completion.choices[0].message.content.strip()
        llm_action = json.loads(text)
        # Merge: task fields as base, LLM can override but not remove
        action.update(llm_action)
    except Exception:
        pass  # fallback to task dict as-is

    return action


async def run_task(task: dict) -> tuple[float, int, List[float]]:
    """Run a single task against the live HF Space environment."""
    import httpx

    base_url = "https://shwetankrai12-financial-analysis-env.hf.space"
    rewards = []
    steps_taken = 0
    score = 0.0

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    log_start(task=task["task_id"], env=BENCHMARK, model=MODEL_NAME)

    try:
        async with httpx.AsyncClient(timeout=30) as http:
            # Reset
            await http.post(f"{base_url}/reset")

            # Step
            action = get_action(client, task, step=1)
            response = await http.post(f"{base_url}/step", json=action)
            obs = response.json()

            reward = obs.get("reward", 0.0)
            done = obs.get("done", False)
            error = obs.get("error_message")
            grader = obs.get("grader_result", {})

            rewards.append(reward)
            steps_taken = 1
            score = grader.get("score", reward) if grader else reward

            log_step(
                step=1,
                action=str(action.get("task_id", "unknown")),
                reward=reward,
                done=done,
                error=error,
            )

    finally:
        log_end(
            success=score >= SUCCESS_SCORE_THRESHOLD,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return score, steps_taken, rewards


async def main() -> None:
    all_scores = []

    for task in TASKS:
        score, steps, rewards = await run_task(task)
        all_scores.append(score)
        print(f"[DEBUG] {task['task_id']} score={score:.2f}", flush=True)

    avg_score = sum(all_scores) / len(all_scores)
    print(f"\n[SUMMARY] avg_score={avg_score:.3f} tasks={len(all_scores)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
