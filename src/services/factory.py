from services.data_loader import DataLoader


class ServiceFactory:
    def __init__(self):
        self._data_loader_service = None

    def get_data_loader_service(self):
        if self._data_loader_service is None:
            self._data_loader_service = DataLoader()
        return self._data_loader_service


service_factory = ServiceFactory()
