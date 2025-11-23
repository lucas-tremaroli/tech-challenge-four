from fastapi import FastAPI
from src.api.routes import router
from src.container import Container


def create_app() -> FastAPI:
    container = Container()
    container.wire(modules=["src.utils.service_factory"])

    app = FastAPI()
    app.container = container
    app.include_router(router)
    return app


app = create_app()
