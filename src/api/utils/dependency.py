from src.container import Container

container = Container()
container.wire(modules=["src.utils.service_factory"])
