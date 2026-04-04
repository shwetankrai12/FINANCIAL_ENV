"""
Comprehensive tests for the Financial Analysis Environment.
Covers environment logic, graders, and HTTP API endpoints.
"""
import pytest
from httpx import AsyncClient, ASGITransport

from server.app import app
from server.environment import FinancialEnvironment, MAX_STEPS
from models.types import (
    FinancialAction,
    FinancialObservation,
    PriceData,
    TechnicalSignals,
    PortfolioEntry,
)
from graders.task_graders import (
    grade_price_check,
    grade_trend_analysis,
    grade_portfolio_rank,
    run_grader,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_price_obs(success=True, price=150.0, low=100.0, high=200.0, volume=1000000, change=-1.5):
    pd = PriceData(
        ticker="TEST",
        current_price=price,
        open_price=148.0,
        period_high=high,
        period_low=low,
        volume=volume,
        daily_change_pct=change,
    ) if success else None
    return FinancialObservation(
        success=success,
        task_id="price_check",
        price_data=pd,
        error_message=None if success else "fail",
    )


def make_trend_obs(rsi=60.0, sma_20=145.0, sma_50=140.0, trend="bullish", success=True):
    sigs = TechnicalSignals(sma_20=sma_20, sma_50=sma_50, rsi_14=rsi, trend=trend) if success else None
    return FinancialObservation(
        success=success,
        task_id="trend_analysis",
        signals=sigs,
        error_message=None if success else "fail",
    )


def make_portfolio_obs(tickers=("AAPL", "MSFT", "GOOGL"), changes=(1.5, -0.5, 0.8), best="AAPL", worst="MSFT"):
    port = [
        PortfolioEntry(ticker=t, current_price=100.0 + i * 10, daily_change_pct=c)
        for i, (t, c) in enumerate(zip(tickers, changes))
    ]
    return FinancialObservation(
        success=True,
        task_id="portfolio_rank",
        portfolio=port,
        best_ticker=best,
        worst_ticker=worst,
    )


# ── Environment unit tests ────────────────────────────────────────────────────

class TestEnvironmentReset:
    def test_reset_returns_success(self):
        env = FinancialEnvironment()
        obs = env.reset()
        assert obs.success is True

    def test_reset_task_id(self):
        env = FinancialEnvironment()
        obs = env.reset()
        assert obs.task_id == "reset"

    def test_reset_done_false(self):
        env = FinancialEnvironment()
        obs = env.reset()
        assert obs.done is False

    def test_reset_clears_step_count(self):
        env = FinancialEnvironment()
        env.reset()
        assert env.state.step_count == 0

    def test_two_resets_give_different_episode_ids(self):
        env = FinancialEnvironment()
        env.reset()
        id1 = env.state.episode_id
        env.reset()
        id2 = env.state.episode_id
        assert id1 != id2


class TestEnvironmentStep:
    def test_unknown_task_returns_failure(self):
        obs = FinancialObservation(
            success=False,
            task_id="bogus_task",
            error_message="Unknown task_id: bogus_task",
        )
        result = run_grader(obs)
        assert result.score == 0.0
        assert result.passed is False

    def test_portfolio_rank_without_tickers_fails(self):
        env = FinancialEnvironment()
        env.reset()
        import asyncio
        action = FinancialAction(task_id="portfolio_rank", ticker="AAPL", tickers=None)
        obs = asyncio.run(env.step_async(action))
        assert obs.success is False

    def test_portfolio_rank_with_one_ticker_fails(self):
        env = FinancialEnvironment()
        env.reset()
        import asyncio
        action = FinancialAction(task_id="portfolio_rank", ticker="AAPL", tickers=["AAPL"])
        obs = asyncio.run(env.step_async(action))
        assert obs.success is False

    def test_episode_done_at_max_steps(self):
        env = FinancialEnvironment()
        env.reset()
        env._state.step_count = MAX_STEPS - 1
        import asyncio
        action = FinancialAction(task_id="price_check", ticker="AAPL")
        obs = asyncio.run(env.step_async(action))
        assert obs.done is True

    def test_reward_is_grader_score(self):
        env = FinancialEnvironment()
        env.reset()
        import asyncio
        action = FinancialAction(task_id="price_check", ticker="AAPL")
        obs = asyncio.run(env.step_async(action))
        assert obs.reward == obs.grader_result.score


# ── Grader unit tests ─────────────────────────────────────────────────────────

class TestGraderScoreRange:
    def test_price_check_score_range_success(self):
        obs = make_price_obs()
        result = grade_price_check(obs)
        assert 0.0 <= result.score <= 1.0

    def test_price_check_score_range_failure(self):
        obs = make_price_obs(success=False)
        result = grade_price_check(obs)
        assert 0.0 <= result.score <= 1.0

    def test_trend_analysis_score_range_success(self):
        obs = make_trend_obs()
        result = grade_trend_analysis(obs)
        assert 0.0 <= result.score <= 1.0

    def test_trend_analysis_score_range_failure(self):
        obs = make_trend_obs(success=False)
        result = grade_trend_analysis(obs)
        assert 0.0 <= result.score <= 1.0

    def test_portfolio_rank_score_range_success(self):
        obs = make_portfolio_obs()
        result = grade_portfolio_rank(obs)
        assert 0.0 <= result.score <= 1.0

    def test_portfolio_rank_score_range_failure(self):
        obs = FinancialObservation(success=False, task_id="portfolio_rank", error_message="fail")
        result = grade_portfolio_rank(obs)
        assert 0.0 <= result.score <= 1.0

    def test_run_grader_unknown_task(self):
        obs = FinancialObservation(success=False, task_id="unknown_xyz")
        result = run_grader(obs)
        assert result.score == 0.0
        assert result.passed is False


class TestGraderPriceCheck:
    def test_full_score(self):
        obs = make_price_obs()
        result = grade_price_check(obs)
        assert result.score == 1.0
        assert result.passed is True

    def test_missing_daily_change(self):
        obs = make_price_obs(change=None)
        result = grade_price_check(obs)
        assert result.score == pytest.approx(0.8)

    def test_price_zero_fails(self):
        obs = make_price_obs(price=0.0, low=0.0)
        result = grade_price_check(obs)
        assert result.score < 0.7


class TestGraderTrendAnalysis:
    def test_bullish_consistent(self):
        obs = make_trend_obs(rsi=60.0, trend="bullish")
        result = grade_trend_analysis(obs)
        assert result.score == 1.0
        assert result.passed is True

    def test_bearish_consistent(self):
        obs = make_trend_obs(rsi=40.0, trend="bearish")
        result = grade_trend_analysis(obs)
        assert result.score == 1.0

    def test_neutral_consistent(self):
        obs = make_trend_obs(rsi=50.0, trend="neutral")
        result = grade_trend_analysis(obs)
        assert result.score == 1.0

    def test_trend_mismatch_loses_points(self):
        obs = make_trend_obs(rsi=60.0, trend="bearish")
        result = grade_trend_analysis(obs)
        assert result.score < 1.0

    def test_missing_sma_loses_points(self):
        obs = make_trend_obs(sma_20=None, sma_50=None)
        result = grade_trend_analysis(obs)
        assert result.score < 1.0


class TestGraderPortfolioRank:
    def test_full_score(self):
        obs = make_portfolio_obs()
        result = grade_portfolio_rank(obs)
        assert result.score == 1.0
        assert result.passed is True

    def test_best_wrong_loses_points(self):
        obs = make_portfolio_obs(best="GOOGL")  # actual best is AAPL (change=1.5)
        result = grade_portfolio_rank(obs)
        assert result.score < 1.0

    def test_best_equals_worst_loses_points(self):
        obs = make_portfolio_obs(best="AAPL", worst="AAPL")
        result = grade_portfolio_rank(obs)
        assert result.score < 1.0

    def test_single_ticker_fails(self):
        port = [PortfolioEntry(ticker="AAPL", current_price=150.0)]
        obs = FinancialObservation(
            success=True, task_id="portfolio_rank",
            portfolio=port, best_ticker="AAPL", worst_ticker="AAPL"
        )
        result = grade_portfolio_rank(obs)
        assert result.score == 0.0
        assert result.passed is False


# ── HTTP API tests ────────────────────────────────────────────────────────────

@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


class TestHTTPEndpoints:
    async def test_health_returns_ok(self, client):
        r = await client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"

    async def test_metadata_has_tasks(self, client):
        r = await client.get("/metadata")
        assert r.status_code == 200
        data = r.json()
        assert "tasks" in data
        assert len(data["tasks"]) == 3

    async def test_metadata_task_ids(self, client):
        r = await client.get("/metadata")
        data = r.json()
        ids = {t["id"] for t in data["tasks"]}
        assert ids == {"price_check", "trend_analysis", "portfolio_rank"}

    async def test_reset_returns_success(self, client):
        r = await client.post("/reset")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True

    async def test_state_has_episode_id(self, client):
        await client.post("/reset")
        r = await client.get("/state")
        assert r.status_code == 200
        data = r.json()
        assert "episode_id" in data
        assert len(data["episode_id"]) > 0

    async def test_state_has_step_count(self, client):
        await client.post("/reset")
        r = await client.get("/state")
        data = r.json()
        assert "step_count" in data
        assert isinstance(data["step_count"], int)
