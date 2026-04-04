"""
Async wrappers around yfinance. All blocking calls run in thread pool.
"""
import asyncio
from functools import partial
from typing import List, Optional
import yfinance as yf

from models.types import PriceData, TechnicalSignals, PortfolioEntry


async def fetch_price(ticker: str, period: str = "3mo") -> PriceData:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_price_sync, ticker, period))


async def fetch_signals(ticker: str, period: str = "3mo") -> TechnicalSignals:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_signals_sync, ticker, period))


async def fetch_portfolio(tickers: List[str], period: str = "3mo") -> List[PortfolioEntry]:
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, partial(_portfolio_entry_sync, t, period)) for t in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    entries = []
    for r in results:
        if isinstance(r, Exception):
            continue
        entries.append(r)
    return entries


def _price_sync(ticker: str, period: str) -> PriceData:
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        raise ValueError(f"No data for {ticker}")
    latest = hist.iloc[-1]
    daily_change = None
    if len(hist) >= 2:
        prev = float(hist["Close"].iloc[-2])
        curr = float(latest["Close"])
        daily_change = round(((curr - prev) / prev) * 100, 2)
    return PriceData(
        ticker=ticker.upper(),
        current_price=round(float(latest["Close"]), 2),
        open_price=round(float(latest["Open"]), 2),
        period_high=round(float(hist["High"].max()), 2),
        period_low=round(float(hist["Low"].min()), 2),
        volume=int(latest["Volume"]),
        daily_change_pct=daily_change,
    )


def _signals_sync(ticker: str, period: str) -> TechnicalSignals:
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        return TechnicalSignals()
    close = hist["Close"]
    sma_20 = _sma(close, 20)
    sma_50 = _sma(close, 50)
    rsi = _rsi(close)
    trend = _trend(float(close.iloc[-1]), sma_20, sma_50, rsi)
    return TechnicalSignals(sma_20=sma_20, sma_50=sma_50, rsi_14=rsi, trend=trend)


def _portfolio_entry_sync(ticker: str, period: str) -> PortfolioEntry:
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        raise ValueError(f"No data for {ticker}")
    latest = hist.iloc[-1]
    close = hist["Close"]
    daily_change = None
    if len(hist) >= 2:
        prev = float(close.iloc[-2])
        curr = float(latest["Close"])
        daily_change = round(((curr - prev) / prev) * 100, 2)
    rsi = _rsi(close)
    sma_20 = _sma(close, 20)
    sma_50 = _sma(close, 50)
    trend = _trend(float(latest["Close"]), sma_20, sma_50, rsi)
    return PortfolioEntry(
        ticker=ticker.upper(),
        current_price=round(float(latest["Close"]), 2),
        daily_change_pct=daily_change,
        rsi_14=rsi,
        trend=trend,
    )


def _sma(series, window: int) -> Optional[float]:
    if len(series) < window:
        return None
    return round(float(series.rolling(window).mean().iloc[-1]), 2)


def _rsi(series, period: int = 14) -> Optional[float]:
    if len(series) < period + 1:
        return None
    delta = series.diff().dropna()
    gain = delta.clip(lower=0).rolling(period).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(period).mean().iloc[-1]
    if loss == 0:
        return 100.0
    return round(float(100 - (100 / (1 + gain / loss))), 2)


def _trend(price: float, sma_20, sma_50, rsi) -> Optional[str]:
    if sma_20 is None or sma_50 is None or rsi is None:
        return "neutral"
    if price > sma_20 > sma_50 and rsi > 55:
        return "bullish"
    if price < sma_20 < sma_50 and rsi < 45:
        return "bearish"
    return "neutral"
