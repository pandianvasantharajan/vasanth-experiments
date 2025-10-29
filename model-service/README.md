# Model Service

A unified FastAPI service that serves both temperature forecasting and object detection models from pickle files.

## Features

- **Temperature Forecasting**: Predict temperature based on temporal features and lag values
- **Object Detection**: Detect objects in uploaded images using YOLOv8-based model
- **Health Monitoring**: Service health checks and model status endpoints
- **CORS Support**: Cross-origin requests enabled for web applications
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation

## API Endpoints

### Core Endpoints

- `GET /` - API information and available endpoints
- `GET /health` - Service health check
- `GET /models` - Model information and status
- `POST /predict/temperature` - Temperature prediction
- `POST /detect/objects` - Object detection in images

### Documentation

- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

## Installation

### Prerequisites

- Python 3.8+
- Poetry (for dependency management)

### Setup

1. **Install dependencies**:
   ```bash
   poetry install
   ```

2. **Activate virtual environment**:
   ```bash
   poetry shell
   ```

3. **Ensure models are available**:
   Make sure the pickle model files exist in `../shared-models/`:
   - `forecasting_model.pkl`
   - `object_detection_model.pkl`
   - `model_registry.json`

## Usage

### Start the Service

```bash
# Development mode with auto-reload
poetry run uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
poetry run uvicorn main:app --host 0.0.0.0 --port 8000
```

The service will be available at `http://localhost:8000`

### API Examples

#### Temperature Prediction

```bash
curl -X POST "http://localhost:8000/predict/temperature" \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 14,
    "month": 7,
    "day_of_year": 195,
    "day_of_week": 3,
    "temp_lag_1": 25.5,
    "temp_lag_6": 23.8,
    "temp_lag_24": 22.1
  }'
```

Response:
```json
{
  "temperature": 26.8,
  "confidence": 0.85,
  "unit": "celsius",
  "prediction_time": "2024-01-15T14:30:00",
  "model_used": "Seasonal_Pattern_Model"
}
```

#### Object Detection

```bash
curl -X POST "http://localhost:8000/detect/objects" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "confidence_threshold=0.5"
```

Response:
```json
{
  "detections": [
    {
      "class_name": "person",
      "class_id": 0,
      "confidence": 0.92,
      "bbox": {
        "x1": 100,
        "y1": 50,
        "x2": 200,
        "y2": 300
      }
    }
  ],
  "num_detections": 1,
  "image_size": [640, 640],
  "inference_time": 0.045,
  "detection_time": "2024-01-15T14:30:00"
}
```

#### Health Check

```bash
curl "http://localhost:8000/health"
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T14:30:00",
  "models": {
    "forecasting": "available",
    "object_detection": "available"
  }
}
```

## Model Information

### Temperature Forecasting Model

- **Algorithm**: Seasonal Pattern Model
- **Input Features**:
  - `hour`: Hour of day (0-23)
  - `month`: Month (1-12)
  - `day_of_year`: Day of year (1-365)
  - `day_of_week`: Day of week (0-6, optional)
  - `temp_lag_1`: Temperature 1 hour ago (optional)
  - `temp_lag_6`: Temperature 6 hours ago (optional)
  - `temp_lag_24`: Temperature 24 hours ago (optional)
- **Output**: Temperature in Celsius with confidence score

### Object Detection Model

- **Algorithm**: YOLOv8n Mock (COCO-based)
- **Classes**: 80 COCO classes (person, car, chair, etc.)
- **Input**: Image files (JPEG, PNG)
- **Output**: Bounding boxes, class names, confidence scores

## Development

### Testing

```bash
# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=main
```

### Code Formatting

```bash
# Format code
poetry run black main.py

# Sort imports
poetry run isort main.py

# Lint code
poetry run flake8 main.py
```

### Adding Dependencies

```bash
# Add production dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name
```

## Deployment

### Docker (Optional)

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY pyproject.toml poetry.lock ./
   RUN pip install poetry && poetry install --no-dev
   
   COPY . .
   
   EXPOSE 8000
   
   CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Build and run**:
   ```bash
   docker build -t model-service .
   docker run -p 8000:8000 model-service
   ```

### Production Considerations

- Use a production ASGI server like Gunicorn with Uvicorn workers
- Set up proper logging and monitoring
- Configure environment variables for sensitive settings
- Use a reverse proxy (nginx) for static files and SSL termination
- Implement proper error handling and rate limiting

## Troubleshooting

### Common Issues

1. **Models not loading**:
   - Check that pickle files exist in `../shared-models/`
   - Verify file permissions
   - Check error logs during startup

2. **Import errors**:
   - Ensure all dependencies are installed: `poetry install`
   - Activate virtual environment: `poetry shell`

3. **Port conflicts**:
   - Change port: `--port 8001`
   - Check for running services: `lsof -i :8000`

### Logs and Debugging

- Enable debug mode: `--log-level debug`
- Check service logs for detailed error information
- Use `/health` endpoint to verify model availability

## License

This project is part of the vasanth-experiments monorepo.