"""
Deterministic graders for each task.
Each grader scores agent output 0.0 to 1.0 based on clear criteria.

Task difficulty:
  price_check    → Easy   (0.0-1.0)
  trend_analysis → Medium (0.0-1.0)
  portfolio_rank → Hard   (0.0-1.0)
"""

from models.types import FinancialObservation, GraderResult


def grade_price_check(obs: FinancialObservation) -> GraderResult:
    """
    Easy task grader.
    Criteria:
      - price > 0                         → +0.4
      - price within period high/low      → +0.3
      - daily_change_pct not None         → +0.2
      - volume > 0                        → +0.1
    Max score: 1.0
    """
    task_id = "price_check"

    if not obs.success or obs.price_data is None:
        return GraderResult(
            task_id=task_id, score=0.0, passed=False,
            reason=f"Task failed: {obs.error_message or 'no price data'}"
        )

    p = obs.price_data
    score = 0.0
    reasons = []

    if p.current_price > 0:
        score += 0.4
        reasons.append("price > 0")
    else:
        reasons.append("FAIL: price <= 0")

    if p.period_low <= p.current_price <= p.period_high:
        score += 0.3
        reasons.append("price within period range")
    else:
        reasons.append("FAIL: price outside period range")

    if p.daily_change_pct is not None:
        score += 0.2
        reasons.append("daily change computed")

    if p.volume > 0:
        score += 0.1
        reasons.append("volume > 0")

    score = round(score, 2)
    return GraderResult(
        task_id=task_id,
        score=score,
        passed=score >= 0.7,
        reason=" | ".join(reasons)
    )


def grade_trend_analysis(obs: FinancialObservation) -> GraderResult:
    """
    Medium task grader.
    Criteria:
      - RSI in valid range 0-100          → +0.3
      - SMA20 and SMA50 present           → +0.2
      - trend label present               → +0.2
      - trend consistent with RSI value   → +0.3
        (RSI > 55 → bullish, RSI < 45 → bearish, else neutral)
    Max score: 1.0
    """
    task_id = "trend_analysis"

    if not obs.success or obs.signals is None:
        return GraderResult(
            task_id=task_id, score=0.0, passed=False,
            reason=f"Task failed: {obs.error_message or 'no signals'}"
        )

    s = obs.signals
    score = 0.0
    reasons = []

    if s.rsi_14 is not None and 0.0 <= s.rsi_14 <= 100.0:
        score += 0.3
        reasons.append(f"RSI valid ({s.rsi_14})")
    else:
        reasons.append("FAIL: RSI missing or out of range")

    if s.sma_20 is not None and s.sma_50 is not None:
        score += 0.2
        reasons.append("SMA20 and SMA50 present")
    else:
        reasons.append("FAIL: SMA missing")

    if s.trend is not None:
        score += 0.2
        reasons.append(f"trend label present ({s.trend})")
    else:
        reasons.append("FAIL: trend label missing")

    # Consistency check: trend must match RSI
    if s.rsi_14 is not None and s.trend is not None:
        expected = (
            "bullish" if s.rsi_14 > 55
            else "bearish" if s.rsi_14 < 45
            else "neutral"
        )
        if s.trend == expected:
            score += 0.3
            reasons.append(f"trend consistent with RSI (expected {expected})")
        else:
            reasons.append(f"FAIL: trend mismatch (got {s.trend}, expected {expected})")

    score = round(score, 2)
    return GraderResult(
        task_id=task_id,
        score=score,
        passed=score >= 0.7,
        reason=" | ".join(reasons)
    )


def grade_portfolio_rank(obs: FinancialObservation) -> GraderResult:
    """
    Hard task grader.
    Criteria:
      - portfolio list present with 2+ entries     → +0.2
      - all entries have valid prices              → +0.2
      - best_ticker identified                     → +0.2
      - worst_ticker identified                    → +0.2
      - best != worst                              → +0.1
      - best_ticker actually has highest change    → +0.1
    Max score: 1.0
    """
    task_id = "portfolio_rank"

    if not obs.success or obs.portfolio is None:
        return GraderResult(
            task_id=task_id, score=0.0, passed=False,
            reason=f"Task failed: {obs.error_message or 'no portfolio data'}"
        )

    port = obs.portfolio
    score = 0.0
    reasons = []

    if len(port) >= 2:
        score += 0.2
        reasons.append(f"{len(port)} tickers in portfolio")
    else:
        reasons.append("FAIL: need at least 2 tickers")
        return GraderResult(task_id=task_id, score=0.0, passed=False, reason=" | ".join(reasons))

    all_valid = all(e.current_price > 0 for e in port)
    if all_valid:
        score += 0.2
        reasons.append("all prices valid")
    else:
        reasons.append("FAIL: some prices invalid")

    if obs.best_ticker:
        score += 0.2
        reasons.append(f"best_ticker identified: {obs.best_ticker}")
    else:
        reasons.append("FAIL: best_ticker missing")

    if obs.worst_ticker:
        score += 0.2
        reasons.append(f"worst_ticker identified: {obs.worst_ticker}")
    else:
        reasons.append("FAIL: worst_ticker missing")

    if obs.best_ticker and obs.worst_ticker and obs.best_ticker != obs.worst_ticker:
        score += 0.1
        reasons.append("best != worst")

    # Verify best_ticker actually has highest daily change
    if obs.best_ticker and all(e.daily_change_pct is not None for e in port):
        actual_best = max(port, key=lambda e: e.daily_change_pct).ticker
        if obs.best_ticker == actual_best:
            score += 0.1
            reasons.append("best_ticker matches highest daily change")
        else:
            reasons.append(f"FAIL: best_ticker wrong (actual best: {actual_best})")

    score = round(score, 2)
    return GraderResult(
        task_id=task_id,
        score=score,
        passed=score >= 0.7,
        reason=" | ".join(reasons)
    )


def run_grader(obs: FinancialObservation) -> GraderResult:
    """Dispatch to correct grader based on task_id."""
    graders = {
        "price_check": grade_price_check,
        "trend_analysis": grade_trend_analysis,
        "portfolio_rank": grade_portfolio_rank,
    }
    fn = graders.get(obs.task_id)
    if fn is None:
        return GraderResult(
            task_id=obs.task_id, score=0.0, passed=False,
            reason=f"Unknown task_id: {obs.task_id}"
        )
    return fn(obs)
