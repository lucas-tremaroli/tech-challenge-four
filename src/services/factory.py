from services.model import ModelService
from services.data_loader import DataLoader
from services.technical_indicators import TechnicalIndicators


class ServiceFactory:
    _data_loader_service = None
    _model_service = None
    _technical_indicators_service = None

    @staticmethod
    def get_data_loader_service():
        if ServiceFactory._data_loader_service is None:
            technical_indicators = ServiceFactory.get_technical_indicators_service()
            ServiceFactory._data_loader_service = DataLoader(technical_indicators)
        return ServiceFactory._data_loader_service

    @staticmethod
    def get_model_service(input_shape, output_units):
        if ServiceFactory._model_service is None:
            ServiceFactory._model_service = ModelService(input_shape, output_units)
        return ServiceFactory._model_service

    @staticmethod
    def get_technical_indicators_service():
        if ServiceFactory._technical_indicators_service is None:
            ServiceFactory._technical_indicators_service = TechnicalIndicators()
        return ServiceFactory._technical_indicators_service
