from dependency_injector import containers, providers
from src.config.data_config import DataConfig
from src.config.model_config import ModelConfig, PlotConfig
from src.config.app_config import IndicatorConfig
from src.services.data.fetcher import YahooFinanceDataFetcher
from src.services.data.repository import DuckDBRepository
from src.services.data.preprocessor import DataPreprocessor
from src.services.indicators.technical_indicators import TechnicalIndicatorsService
from src.services.ml.model_builder import LSTMModelBuilder
from src.services.ml.trainer import ModelTrainer
from src.services.ml.evaluator import ModelEvaluator


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=["src.utils.service_factory"]
    )

    data_config = providers.Factory(DataConfig)
    model_config = providers.Factory(ModelConfig)
    plot_config = providers.Factory(PlotConfig)
    indicator_config = providers.Factory(IndicatorConfig)

    data_fetcher = providers.Factory(YahooFinanceDataFetcher)

    data_repository = providers.Factory(DuckDBRepository, config=data_config)

    data_preprocessor = providers.Factory(DataPreprocessor, config=data_config)

    technical_indicators_service = providers.Factory(
        TechnicalIndicatorsService, config=indicator_config
    )

    model_builder = providers.Factory(LSTMModelBuilder, config=model_config)

    model_trainer = providers.Factory(
        ModelTrainer, config=model_config, model_builder=model_builder
    )

    model_evaluator = providers.Factory(
        ModelEvaluator, model_config=model_config, plot_config=plot_config
    )
