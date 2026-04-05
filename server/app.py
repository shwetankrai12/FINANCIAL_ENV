"""
OpenEnv FastAPI server.
Endpoints: POST /reset, POST /step, GET /state, WS /ws, GET /health, GET /metadata
"""
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from models.types import FinancialAction, FinancialObservation, FinancialState
from server.environment import FinancialEnvironment

_shared_env = FinancialEnvironment()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Financial Analysis Environment",
        description="OpenEnv RL environment — 3 graded financial tasks (easy/medium/hard)",
        version="1.0.0",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    async def health():
        return {"status": "ok", "env": "financial-analysis-env"}

    @app.get("/metadata")
    async def metadata():
        return {
            "name": "financial-analysis-env",
            "version": "1.0.0",
            "tasks": [
                {"id": "price_check", "difficulty": "easy", "reward_range": [0.0, 1.0]},
                {"id": "trend_analysis", "difficulty": "medium", "reward_range": [0.0, 1.0]},
                {"id": "portfolio_rank", "difficulty": "hard", "reward_range": [0.0, 1.0]},
            ],
            "max_steps": 10,
        }

    @app.post("/reset", response_model=FinancialObservation)
    async def reset():
        return await _shared_env.reset_async()

    @app.post("/step", response_model=FinancialObservation)
    async def step(action: FinancialAction):
        return await _shared_env.step_async(action)

    @app.get("/state", response_model=FinancialState)
    async def state():
        return _shared_env.state

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        env = FinancialEnvironment()
        try:
            while True:
                msg = json.loads(await websocket.receive_text())
                a = msg.get("action", "")
                if a == "reset":
                    obs = await env.reset_async()
                    await websocket.send_text(obs.model_dump_json())
                elif a == "state":
                    await websocket.send_text(env.state.model_dump_json())
                elif a == "step":
                    obs = await env.step_async(FinancialAction(**msg.get("payload", {})))
                    await websocket.send_text(obs.model_dump_json())
                else:
                    await websocket.send_text(json.dumps({"error": f"unknown action: {a}"}))
        except WebSocketDisconnect:
            pass

    return app


app = create_app()


def main():
    import uvicorn
    import os
    port = int(os.getenv("APP_PORT", 7860))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )

if __name__ == "__main__":
    main()
