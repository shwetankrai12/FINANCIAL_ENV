from typing import Optional, Literal, List, Dict
from pydantic import BaseModel, Field


class FinancialAction(BaseModel):
    task_id: Literal["price_check", "trend_analysis", "portfolio_rank"] = Field(
        ..., description="Task to run: price_check (easy), trend_analysis (medium), portfolio_rank (hard)"
    )
    ticker: str = Field(..., min_length=1, max_length=12, description="Primary ticker e.g. AAPL")
    period: str = Field(default="3mo", description="yfinance period: 1d 5d 1mo 3mo 6mo 1y")
    tickers: Optional[List[str]] = Field(default=None, description="For portfolio_rank: list of 2-5 tickers")


class PriceData(BaseModel):
    ticker: str
    current_price: float
    open_price: float
    period_high: float
    period_low: float
    volume: int
    daily_change_pct: Optional[float] = None


class TechnicalSignals(BaseModel):
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    rsi_14: Optional[float] = None
    trend: Optional[Literal["bullish", "bearish", "neutral"]] = None


class PortfolioEntry(BaseModel):
    ticker: str
    current_price: float
    daily_change_pct: Optional[float] = None
    rsi_14: Optional[float] = None
    trend: Optional[str] = None


class GraderResult(BaseModel):
    task_id: str
    score: float = Field(..., ge=0.0, le=1.0, description="Grader score 0.0 to 1.0")
    passed: bool
    reason: str


class FinancialObservation(BaseModel):
    success: bool
    task_id: str
    price_data: Optional[PriceData] = None
    signals: Optional[TechnicalSignals] = None
    portfolio: Optional[List[PortfolioEntry]] = None
    best_ticker: Optional[str] = None
    worst_ticker: Optional[str] = None
    llm_insight: Optional[str] = None
    grader_result: Optional[GraderResult] = None
    error_message: Optional[str] = None
    reward: float = 0.0
    done: bool = False


class FinancialState(BaseModel):
    episode_id: str
    step_count: int
    scores: Dict[str, float] = {}
    last_task: Optional[str] = None
