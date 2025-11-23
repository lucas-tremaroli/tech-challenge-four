from services.model import ModelService
from services.data_loader import DataLoaderService
from services.technical_indicators import TechnicalIndicatorsService


class ServiceFactory:
    _data_loader_service: DataLoaderService = None
    _model_service: ModelService = None
    _technical_indicators_service: TechnicalIndicatorsService = None

    @staticmethod
    def get_data_loader_service() -> DataLoaderService:
        if ServiceFactory._data_loader_service is None:
            technical_indicators = ServiceFactory.get_technical_indicators_service()
            ServiceFactory._data_loader_service = DataLoaderService(
                technical_indicators
            )
        return ServiceFactory._data_loader_service

    @staticmethod
    def get_model_service() -> ModelService:
        if ServiceFactory._model_service is None:
            ServiceFactory._model_service = ModelService()
        return ServiceFactory._model_service

    @staticmethod
    def get_technical_indicators_service() -> TechnicalIndicatorsService:
        if ServiceFactory._technical_indicators_service is None:
            ServiceFactory._technical_indicators_service = TechnicalIndicatorsService()
        return ServiceFactory._technical_indicators_service
