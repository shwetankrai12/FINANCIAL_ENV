import os
from typing import Optional
from huggingface_hub import InferenceClient

from models.types import PriceData, TechnicalSignals


def get_llm_insight(
    price: PriceData,
    signals: TechnicalSignals,
    query: Optional[str] = None,
) -> str:
    """
    Calls HF InferenceClient (Llama-3.3-70B-Instruct) synchronously.
    Returns concise analysis string. Falls back gracefully if HF_TOKEN missing.
    """
    token = os.getenv("HF_TOKEN", "")
    if not token:
        return "LLM insight unavailable: HF_TOKEN not set."

    client = InferenceClient(token=token)

    context = (
        f"Ticker: {price.ticker}\n"
        f"Price: ${price.current_price} | Daily Change: {price.daily_change_pct}%\n"
        f"Period High: ${price.period_high} | Low: ${price.period_low}\n"
        f"SMA20: {signals.sma_20} | SMA50: {signals.sma_50} | RSI14: {signals.rsi_14}\n"
        f"Trend: {signals.trend}"
    )
    user_q = query or "Give a brief technical analysis and market outlook in 3 sentences."

    response = client.chat_completion(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {
                "role": "system",
                "content": "You are a concise financial analyst. Keep responses under 100 words. Be factual.",
            },
            {"role": "user", "content": f"{context}\n\n{user_q}"},
        ],
        max_tokens=200,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
