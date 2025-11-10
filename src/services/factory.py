from services.model import ModelService
from services.data_loader import DataLoader


class ServiceFactory:
    _data_loader_service = None
    _model_service = None

    @staticmethod
    def get_data_loader_service():
        if ServiceFactory._data_loader_service is None:
            ServiceFactory._data_loader_service = DataLoader()
        return ServiceFactory._data_loader_service

    @staticmethod
    def get_model_service(input_shape, output_units):
        if ServiceFactory._model_service is None:
            ServiceFactory._model_service = ModelService(input_shape, output_units)
        return ServiceFactory._model_service
