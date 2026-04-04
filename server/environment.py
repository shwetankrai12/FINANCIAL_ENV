"""
FinancialEnvironment — full OpenEnv spec implementation.

3 graded tasks:
  price_check    (easy)   → reward = grader score
  trend_analysis (medium) → reward = grader score
  portfolio_rank (hard)   → reward = grader score

Episode ends after MAX_STEPS steps.
Partial reward: grader score flows directly as reward signal.
"""

import asyncio
from uuid import uuid4

from models.types import FinancialAction, FinancialObservation, FinancialState
from server.tools import fetch_price, fetch_signals, fetch_portfolio
from server.llm_agent import get_llm_insight
from graders.task_graders import run_grader

MAX_STEPS = 10


class FinancialEnvironment:

    def __init__(self):
        self._state = FinancialState(episode_id=str(uuid4()), step_count=0)

    # ── OpenEnv required interface ───────────────────────────────

    def reset(self) -> FinancialObservation:
        self._state = FinancialState(episode_id=str(uuid4()), step_count=0)
        return FinancialObservation(
            success=True, task_id="reset", reward=0.0, done=False
        )

    async def reset_async(self) -> FinancialObservation:
        return self.reset()

    def step(self, action: FinancialAction) -> FinancialObservation:
        return asyncio.get_event_loop().run_until_complete(self.step_async(action))

    async def step_async(self, action: FinancialAction) -> FinancialObservation:
        self._state.step_count += 1
        self._state.last_task = action.task_id
        done = self._state.step_count >= MAX_STEPS

        try:
            if action.task_id == "price_check":
                obs = await self._price_check(action)
            elif action.task_id == "trend_analysis":
                obs = await self._trend_analysis(action)
            elif action.task_id == "portfolio_rank":
                obs = await self._portfolio_rank(action)
            else:
                obs = FinancialObservation(
                    success=False, task_id=action.task_id,
                    error_message=f"Unknown task_id: {action.task_id}",
                    reward=0.0,
                )
        except ValueError as e:
            obs = FinancialObservation(
                success=False, task_id=action.task_id,
                error_message=str(e), reward=0.0,
            )
        except Exception as e:
            obs = FinancialObservation(
                success=False, task_id=action.task_id,
                error_message=f"Unexpected error: {e}", reward=0.0,
            )

        # Run grader and assign reward from grader score
        grader_result = run_grader(obs)
        obs.grader_result = grader_result
        obs.reward = grader_result.score
        obs.done = done

        # Track scores per task
        self._state.scores[action.task_id] = grader_result.score

        return obs

    @property
    def state(self) -> FinancialState:
        return self._state

    # ── Task handlers ────────────────────────────────────────────

    async def _price_check(self, action: FinancialAction) -> FinancialObservation:
        price = await fetch_price(action.ticker, action.period)
        return FinancialObservation(
            success=True, task_id="price_check", price_data=price
        )

    async def _trend_analysis(self, action: FinancialAction) -> FinancialObservation:
        price, signals = await asyncio.gather(
            fetch_price(action.ticker, action.period),
            fetch_signals(action.ticker, action.period),
        )
        # LLM insight — non-blocking failure
        llm = None
        try:
            loop = asyncio.get_event_loop()
            llm = await loop.run_in_executor(
                None, lambda: get_llm_insight(price, signals)
            )
        except Exception:
            pass
        return FinancialObservation(
            success=True, task_id="trend_analysis",
            price_data=price, signals=signals, llm_insight=llm
        )

    async def _portfolio_rank(self, action: FinancialAction) -> FinancialObservation:
        tickers = action.tickers
        if not tickers or len(tickers) < 2:
            raise ValueError("portfolio_rank requires tickers list with at least 2 items")
        if len(tickers) > 5:
            tickers = tickers[:5]

        portfolio = await fetch_portfolio(tickers, action.period)
        if len(portfolio) < 2:
            raise ValueError("Could not fetch data for enough tickers")

        # Rank by daily change
        ranked = sorted(
            [e for e in portfolio if e.daily_change_pct is not None],
            key=lambda e: e.daily_change_pct,
            reverse=True,
        )
        best = ranked[0].ticker if ranked else None
        worst = ranked[-1].ticker if ranked else None

        return FinancialObservation(
            success=True, task_id="portfolio_rank",
            portfolio=portfolio,
            best_ticker=best,
            worst_ticker=worst,
        )
