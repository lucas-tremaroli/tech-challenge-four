# Tech Challenge Four - Stock Price Prediction

A machine learning application that uses LSTM neural networks to predict Apple (AAPL) stock prices with technical indicators.

## Quick Start

1. Install make

    ```bash
    brew install make
    ```

2. Run the following command to start the application:

    ```bash
    make up
    ```

3. Access the application in your web browser at `http://localhost:8000`.

4. Copy the example payload from `assets/payload.json` and send a POST request to test stock price predictions:

    ```bash
    curl -X POST "http://localhost:8000/api/predict" \
          -H "Content-Type: application/json" \
          -d @assets/payload.json
    ```

5. Track system metrics and API performance through the following endpoints:

    - Metrics Dashboard: `http://localhost:8000/metrics`
    - Prometheus Metrics: `http://localhost:8000/prometheus`
    - API Endpoints: `GET /metrics` and `GET /prometheus` for programmatic access

## Model Architecture

**LSTM Neural Network** with the following structure:

- Input layer: `(30, 12)` - 30 time steps with 12 features
- LSTM layer: 16 units with L2 regularization (0.005)
- Dropout layer: 30% dropout rate
- Dense output layer: 12 units (multi-step ahead prediction)
- Optimizer: Adam with MSE loss and MAE metrics

## Hyperparameters

```python
# Training Configuration
epochs = 50
batch_size = 16
learning_rate = 0.001
validation_split = 0.2

# Regularization
dropout_rate = 0.3
l2_regularization = 0.005

# Callbacks
early_stopping_patience = 7
reduce_lr_patience = 3
reduce_lr_factor = 0.2
min_learning_rate = 1e-6

# Data
lookback_period = 30 days
test_size = 0.2
cv_splits = 5 (time series cross-validation)
```

## Training Process

1. **Data Preparation**:

   - Load AAPL stock data from local database (duckdb)
   - Add technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
   - Scale features using MinMaxScaler
   - Create sequences with 30-day lookback window

2. **Validation**:

   - Time series cross-validation (5-fold)
   - Data leakage detection
   - Overfitting monitoring

3. **Training**:

   - Early stopping on validation loss
   - Learning rate reduction on plateau
   - No shuffling (preserves temporal order)

4. **Evaluation**:

   - Test set evaluation
   - Training history plots
   - Prediction vs actual visualization

## API Endpoints

- **`GET /api/health`** - Health check endpoint
- **`POST /api/predict`** - Stock price prediction endpoint
- **`GET /metrics`** - Prometheus metrics for monitoring
- **`GET /prometheus`** - FastAPI instrumentator metrics

## Monitoring & Metrics

The API includes comprehensive monitoring with Prometheus metrics:

### Available Metrics

- **`prediction_requests_total`** - Total prediction requests (with success/error labels)
- **`prediction_duration_seconds`** - End-to-end prediction response times
- **`model_inference_duration_seconds`** - Model inference timing
- **`prediction_steps_total`** - Number of prediction steps processed
- **`prediction_sequence_length`** - Input sequence lengths
- **`memory_usage_bytes`** - Current memory usage
- **`cpu_usage_percent`** - CPU utilization
- **`model_info`** - Model metadata and version information

### Monitoring Setup

1. **View Metrics**: Visit `http://localhost:8000/metrics`
2. **Prometheus Integration**: Configure Prometheus to scrape the `/metrics` endpoint (future improvement)
3. **Grafana Dashboards**: Create dashboards using the exposed metrics (future improvement)
4. **Alerting**: Set up alerts on response times, error rates, and resource usage (future improvement)

## Key Features

- **Time Series CV**: Proper temporal validation without data leakage
- **Technical Indicators**: 12+ financial indicators for enhanced prediction
- **Regularization**: Dropout and L2 to prevent overfitting
- **Monitoring**: Automated overfitting detection and performance tracking
- **API**: FastAPI endpoints for model inference with Prometheus metrics
- **Production Monitoring**: Response time and resource usage
