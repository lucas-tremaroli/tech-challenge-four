import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS


class DBClient:
    def __init__(self, url: str, token: str, org: str, bucket: str):
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.client = influxdb_client.InfluxDBClient(
            url=self.url,
            token=self.token,
            org=self.org
        )
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

    def write_data(self, data):
        self.write_api.write(bucket=self.bucket, org=self.org, record=data)

    def query_data(self, query: str):
        query_api = self.client.query_api()
        return query_api.query(org=self.org, query=query)
