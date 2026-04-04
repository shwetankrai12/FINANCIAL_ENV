import uvicorn
import os

if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", 8000))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("APP_ENV", "development") == "development",
        log_level="info",
    )
