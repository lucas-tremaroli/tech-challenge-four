from services.model import ModelService
from services.data_loader import DataLoader


class ServiceFactory:
    def __init__(self):
        self._data_loader_service = None
        self._model_service = None

    def get_data_loader_service(self):
        if self._data_loader_service is None:
            self._data_loader_service = DataLoader()
        return self._data_loader_service

    def get_model_service(self, input_shape, output_units):
        if self._model_service is None:
            self._model_service = ModelService(input_shape, output_units)
        return self._model_service


service_factory = ServiceFactory()
