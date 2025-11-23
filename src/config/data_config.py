from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    db_file: str = "src/data/aapl.db"
    db_table: str = "aapl_stock_data"
    exclude_columns: List[str] = None
    test_size: float = 0.2
    random_state: int = 42
    lookback_period: int = 30

    def __post_init__(self):
        if self.exclude_columns is None:
            self.exclude_columns = ["Date", "Dividends", "Stock Splits"]