import duckdb
import logging
from pandas import DataFrame
from src.services.interfaces.data_service import IDataRepository
from src.config.data_config import DataConfig


class DuckDBRepository(IDataRepository):
    def __init__(self, config: DataConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config

    def load_data(self) -> DataFrame:
        self.logger.info(
            f"Loading data from {self.config.db_file}, table: {self.config.db_table}"
        )
        try:
            with duckdb.connect(self.config.db_file) as db_con:
                data = db_con.table(self.config.db_table).to_df()
            self.logger.info(f"Successfully loaded {len(data)} records")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise

    def save_data(self, data: DataFrame) -> None:
        self.logger.info(
            f"Saving {len(data)} records to {self.config.db_file}, table: {self.config.db_table}"
        )
        try:
            with duckdb.connect(self.config.db_file) as db_con:
                db_con.register("temp_df", data)
                db_con.execute(
                    f"CREATE OR REPLACE TABLE {self.config.db_table} AS SELECT * FROM temp_df"
                )
            self.logger.info("Data saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save data: {str(e)}")
            raise
