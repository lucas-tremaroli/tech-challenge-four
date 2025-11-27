from fastapi import FastAPI, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
from src.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Stock Prediction API",
        description="ML API for AAPL stock price predictions with monitoring",
        version="1.0.0"
    )
    
    app.include_router(router)
    
    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app, endpoint="/prometheus")
    
    return app


app = create_app()
