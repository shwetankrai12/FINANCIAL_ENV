"""
FinancialEnv client — OpenEnv-compatible.

Usage (async):
    async with FinancialEnv(base_url="ws://localhost:8000") as env:
        result = await env.reset()
        result = await env.step(FinancialAction(command="analyze", ticker="AAPL"))
        print(result.observation.llm_insight)
        print(result.reward)

Usage (sync):
    with FinancialEnv(base_url="ws://localhost:8000").sync() as env:
        result = env.reset()
        result = env.step(FinancialAction(command="get_price", ticker="RELIANCE.NS"))
"""

import json
import asyncio
import websockets
from typing import Optional, AsyncIterator
from contextlib import asynccontextmanager

from models.types import FinancialAction, FinancialObservation, FinancialState


class StepResult:
    """Wraps observation with reward and done flag for RL training loops."""
    def __init__(self, observation: FinancialObservation):
        self.observation = observation
        self.reward = observation.reward
        self.done = observation.done

    def __repr__(self):
        return (
            f"StepResult(ticker={self.observation.price_data.ticker if self.observation.price_data else 'N/A'}, "
            f"reward={self.reward}, done={self.done})"
        )


class FinancialEnv:
    """
    Async WebSocket client for the Financial Analysis Environment.
    Follows the OpenEnv EnvClient interface pattern.
    """

    def __init__(self, base_url: str = "ws://localhost:8000"):
        # Normalise: accept http:// or ws:// prefix
        self._base_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

    # ── Context manager ──────────────────────────────────────────

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    # ── Connection ───────────────────────────────────────────────

    async def connect(self):
        ws_url = f"{self._base_url}/ws"
        self._ws = await websockets.connect(ws_url)

    async def close(self):
        if self._ws:
            await self._ws.close()
            self._ws = None

    def sync(self) -> "SyncFinancialEnv":
        return SyncFinancialEnv(self)

    # ── OpenEnv interface ────────────────────────────────────────

    async def reset(self) -> StepResult:
        """Reset the environment, start a new episode."""
        self._ensure_connected()
        await self._ws.send(json.dumps({"action": "reset"}))
        raw = await self._ws.recv()
        obs = FinancialObservation.model_validate_json(raw)
        return StepResult(obs)

    async def step(self, action: FinancialAction) -> StepResult:
        """Take one step — send action, receive observation + reward."""
        self._ensure_connected()
        payload = {
            "action": "step",
            "payload": action.model_dump(),
        }
        await self._ws.send(json.dumps(payload))
        raw = await self._ws.recv()
        obs = FinancialObservation.model_validate_json(raw)
        return StepResult(obs)

    async def state(self) -> FinancialState:
        """Get current episode state (id, step count, etc.)."""
        self._ensure_connected()
        await self._ws.send(json.dumps({"action": "state"}))
        raw = await self._ws.recv()
        return FinancialState.model_validate_json(raw)

    def _ensure_connected(self):
        if not self._ws:
            raise RuntimeError("Not connected. Use 'async with FinancialEnv(...) as env'")

    # ── Factory methods (OpenEnv pattern) ────────────────────────

    @classmethod
    def from_local(cls, port: int = 8000) -> "FinancialEnv":
        return cls(base_url=f"ws://localhost:{port}")

    @classmethod
    def from_hub(cls, space_id: str) -> "FinancialEnv":
        """Connect to a deployed HF Space, e.g. 'shwetankrai12/financial-env'"""
        slug = space_id.replace("/", "-").lower()
        return cls(base_url=f"wss://{slug}.hf.space")


class SyncFinancialEnv:
    """Synchronous wrapper — for REPL / non-async training loops."""

    def __init__(self, async_client: FinancialEnv):
        self._client = async_client
        self._loop = asyncio.new_event_loop()

    def __enter__(self):
        self._loop.run_until_complete(self._client.connect())
        return self

    def __exit__(self, *args):
        self._loop.run_until_complete(self._client.close())
        self._loop.close()

    def reset(self) -> StepResult:
        return self._loop.run_until_complete(self._client.reset())

    def step(self, action: FinancialAction) -> StepResult:
        return self._loop.run_until_complete(self._client.step(action))

    def state(self) -> FinancialState:
        return self._loop.run_until_complete(self._client.state())