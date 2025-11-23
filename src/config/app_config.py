from dataclasses import dataclass


@dataclass
class AppConfig:
    debug: bool = False
    log_level: str = "INFO"
    api_prefix: str = "/api"


@dataclass
class IndicatorConfig:
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0