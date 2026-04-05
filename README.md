Rewrite README.md completely. Make it professional and impressive for hackathon judges.

Use this exact structure:

---
title: Financial Analysis Environment
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# 📈 Financial Analysis Environment
> OpenEnv-compatible RL Environment — Meta × Hugging Face × PyTorch Hackathon

A production-ready reinforcement learning environment for financial market analysis. 
Agents interact with real-time stock data, compute technical indicators, and earn 
rewards for successful analysis tasks.

---

## 🎯 Why This Environment?

Training RL agents on financial tasks is hard because:
- Market data is noisy and non-stationary
- Reward signals are sparse in real trading
- No standardized environment exists for financial RL

This environment solves all three by providing:
- Real-time data via yfinance (NSE + NYSE stocks)
- Deterministic graders with partial progress rewards
- 3 tasks of increasing difficulty (easy → medium → hard)

---

## 🗂️ Tasks

| Task | Difficulty | Description | Max Score |
|------|-----------|-------------|-----------|
| `price_check` | 🟢 Easy | Fetch & validate OHLCV stock data | 1.0 |
| `trend_analysis` | 🟡 Medium | Compute RSI + SMA + classify trend | 1.0 |
| `portfolio_rank` | 🔴 Hard | Rank N stocks by performance | 1.0 |

---

## 🏆 Grader Criteria

### price_check (Easy)
| Criterion | Points |
|-----------|--------|
| price > 0 | +0.4 |
| price within period high/low | +0.3 |
| daily_change_pct computed | +0.2 |
| volume > 0 | +0.1 |

### trend_analysis (Medium)
| Criterion | Points |
|-----------|--------|
| RSI in valid range (0-100) | +0.3 |
| SMA20 and SMA50 present | +0.2 |
| trend label present | +0.2 |
| trend consistent with RSI | +0.3 |

### portfolio_rank (Hard)
| Criterion | Points |
|-----------|--------|
| 2+ tickers in portfolio | +0.2 |
| all prices valid | +0.2 |
| best_ticker identified | +0.2 |
| worst_ticker identified | +0.2 |
| best != worst | +0.1 |
| best matches highest daily change | +0.1 |

---

## 📊 Baseline Scores

Tested with AAPL (3mo period), portfolio: [AAPL, MSFT, GOOGL]

| Task | Score |
|------|-------|
| price_check (easy) | 1.00 |
| trend_analysis (medium) | 0.70 |
| portfolio_rank (hard) | 1.00 |
| **Average** | **0.90** |

> trend_analysis scores 0.70 by design — partial credit when RSI signals bullish 
> but price/SMA alignment hasn't confirmed the trend. This is meaningful partial 
> reward, not a bug.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset` | Start new episode |
| POST | `/step` | Send action → observation + reward |
| GET | `/state` | Episode metadata |
| WS | `/ws` | Persistent WebSocket session |
| GET | `/health` | Health check |
| GET | `/metadata` | Environment description |
| GET | `/docs` | Swagger UI |

**Live URL:** `https://shwetankrai12-financial-analysis-env.hf.space`

---

## ⚡ Quickstart
```bash
git clone https://github.com/shwetankrai12/FINANCIAL_ENV
cd FINANCIAL_ENV
pip install -r requirements.txt
cp .env.example .env
# Add your HF_TOKEN to .env
python main.py
# Open http://localhost:8000/docs
```

---

## 🤖 Baseline Agent
```bash
python baseline_agent.py
```

Expected output:
price_check:    1.00
trend_analysis: 0.70
portfolio_rank: 1.00
AVERAGE:        0.90

---

## 📦 Action Schema
```json
{
  "task_id": "trend_analysis",
  "ticker": "AAPL",
  "period": "3mo",
  "tickers": null,
  "query": null
}
```

Supported periods: `1d` `5d` `1mo` `3mo` `6mo` `1y`
Supported task_ids: `price_check` `trend_analysis` `portfolio_rank`

---

## 🌏 Indian Stocks Support

Works with NSE tickers too:
```json
{
  "task_id": "portfolio_rank",
  "ticker": "RELIANCE.NS",
  "tickers": ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
  "period": "1mo"
}
```

---

## 🛠️ Stack

| Layer | Technology |
|-------|-----------|
| Environment server | FastAPI + WebSocket |
| Data models | Pydantic v2 |
| Stock data | yfinance (NSE + NYSE) |
| LLM insights | HF InferenceClient (Llama-3.3-70B) |
| Graders | Deterministic, reproducible |
| Containerization | Docker (non-root) |
| Deployment | Hugging Face Spaces |

---

## 🔁 Reward Design

- **Partial rewards** — every grader criterion contributes independently
- **No binary end-of-episode** — agents get signal on every step
- **Deterministic** — same input always gives same score
- **Range** — 0.0 to 1.0 per task

---

After writing README.md, commit and push to both remotes:
git add README.md
git commit -m "docs: comprehensive README for hackathon submission"
git push origin main
git push space main