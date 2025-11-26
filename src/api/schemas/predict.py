from typing import List
from pydantic import BaseModel


class DataPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class PredictionPayload(BaseModel):
    sequence: List[DataPoint]
    steps: int
