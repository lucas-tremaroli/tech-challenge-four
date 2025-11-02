import os
from data.db.client import DBClient


class DependencyContainer:

    def get_db_client(self) -> DBClient:
        return DBClient(
            url="http://localhost:8181",
            token=os.getenv("INFLUXDB_TOKEN", "my-influxdb-token"),
            org="fiap",
            bucket="fiap-bucket"
        )

dependency_container = DependencyContainer()
