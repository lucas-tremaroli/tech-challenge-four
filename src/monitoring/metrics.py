import time
import psutil
import asyncio
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge, Info
from typing import Callable, Any

prediction_requests_total = Counter(
    'prediction_requests_total',
    'Total number of prediction requests',
    ['status']
)

prediction_duration_seconds = Histogram(
    'prediction_duration_seconds',
    'Time spent processing predictions',
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

model_inference_duration_seconds = Histogram(
    'model_inference_duration_seconds',
    'Time spent in model inference',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)

prediction_steps_total = Counter(
    'prediction_steps_total',
    'Total number of prediction steps processed'
)

sequence_length_gauge = Gauge(
    'prediction_sequence_length',
    'Length of input sequences for predictions'
)

memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    'Current memory usage in bytes'
)

cpu_usage_percent = Gauge(
    'cpu_usage_percent',
    'Current CPU usage percentage'
)

model_info = Info(
    'model_info',
    'Information about the loaded model'
)


def track_prediction_metrics(func: Callable) -> Callable:
    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise e
            finally:
                duration = time.time() - start_time
                prediction_duration_seconds.observe(duration)
                prediction_requests_total.labels(status=status).inc()
                
                update_system_metrics()
                
        return async_wrapper
    else:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise e
            finally:
                duration = time.time() - start_time
                prediction_duration_seconds.observe(duration)
                prediction_requests_total.labels(status=status).inc()
                
                update_system_metrics()
                
        return wrapper


def track_model_inference(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            model_inference_duration_seconds.observe(duration)
            
    return wrapper


def update_system_metrics():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage_bytes.set(memory_info.rss)
    
    cpu_percent = psutil.cpu_percent()
    cpu_usage_percent.set(cpu_percent)


def track_prediction_steps(steps: int):
    prediction_steps_total.inc(steps)


def track_sequence_length(length: int):
    sequence_length_gauge.set(length)


def set_model_info(model_path: str, model_type: str = "LSTM"):
    model_info.info({
        'model_path': model_path,
        'model_type': model_type,
        'version': '1.0'
    })