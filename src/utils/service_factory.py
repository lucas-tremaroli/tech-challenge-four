from dependency_injector.wiring import Provide, inject
from src.container import Container
from src.services.interfaces.data_service import (
    IDataFetcher,
    IDataRepository,
    IDataPreprocessor,
)
from src.services.interfaces.indicator_service import IIndicatorService
from src.services.interfaces.model_service import (
    IModelBuilder,
    IModelTrainer,
    IModelEvaluator,
)


class ServiceFactory:
    @staticmethod
    @inject
    def get_data_fetcher(
        fetcher: IDataFetcher = Provide[Container.data_fetcher],
    ) -> IDataFetcher:
        return fetcher

    @staticmethod
    @inject
    def get_data_repository(
        repository: IDataRepository = Provide[Container.data_repository],
    ) -> IDataRepository:
        return repository

    @staticmethod
    @inject
    def get_data_preprocessor(
        preprocessor: IDataPreprocessor = Provide[Container.data_preprocessor],
    ) -> IDataPreprocessor:
        return preprocessor

    @staticmethod
    @inject
    def get_technical_indicators_service(
        service: IIndicatorService = Provide[Container.technical_indicators_service],
    ) -> IIndicatorService:
        return service

    @staticmethod
    @inject
    def get_model_builder(
        builder: IModelBuilder = Provide[Container.model_builder],
    ) -> IModelBuilder:
        return builder

    @staticmethod
    @inject
    def get_model_trainer(
        trainer: IModelTrainer = Provide[Container.model_trainer],
    ) -> IModelTrainer:
        return trainer

    @staticmethod
    @inject
    def get_model_evaluator(
        evaluator: IModelEvaluator = Provide[Container.model_evaluator],
    ) -> IModelEvaluator:
        return evaluator
