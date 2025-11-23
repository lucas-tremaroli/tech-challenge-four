from dataclasses import dataclass


@dataclass
class ModelConfig:
    input_shape: tuple = (30, 12)
    output_units: int = 12
    lstm_units: int = 16
    dropout_rate: float = 0.3
    l2_regularization: float = 0.005
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 16
    validation_split: float = 0.2
    early_stopping_patience: int = 7
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.2
    min_learning_rate: float = 1e-6
    cv_splits: int = 5


@dataclass
class PlotConfig:
    plots_dir: str = "./assets/plots"
    training_history_filename: str = "training_history.png"
    predictions_filename: str = "predictions_vs_actual.png"
    metrics_filename: str = "model_metrics.png"
    dpi: int = 300
