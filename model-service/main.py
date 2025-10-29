from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pickle
import numpy as np
import json
from datetime import datetime
import os
from pathlib import Path
import io
import base64

# Initialize FastAPI app
app = FastAPI(
    title="Model Service API",
    description="Unified API service for temperature forecasting and object detection models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to shared models
MODELS_PATH = Path("/Users/vasantharajanpandian/my-development/zero-development/vasanth-experiments/shared-models")

# Request/Response Models
class TemperaturePredictionRequest(BaseModel):
    hour: int  # 0-23
    month: int  # 1-12
    day_of_year: int  # 1-365
    day_of_week: Optional[int] = None  # 0-6
    temp_lag_1: Optional[float] = None  # Previous hour temperature
    temp_lag_6: Optional[float] = None  # 6 hours ago temperature
    temp_lag_24: Optional[float] = None  # 24 hours ago temperature
    
    class Config:
        schema_extra = {
            "example": {
                "hour": 14,
                "month": 7,
                "day_of_year": 195,
                "day_of_week": 3,
                "temp_lag_1": 25.5,
                "temp_lag_6": 23.8,
                "temp_lag_24": 22.1
            }
        }

class TemperaturePredictionResponse(BaseModel):
    temperature: float
    confidence: float
    unit: str
    prediction_time: str
    model_used: str

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class Detection(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    bbox: BoundingBox

class ObjectDetectionResponse(BaseModel):
    detections: List[Detection]
    num_detections: int
    image_size: List[int]
    inference_time: float
    detection_time: str

class ModelInfo(BaseModel):
    model_type: str
    algorithm: str
    version: str
    status: str

class ModelsStatusResponse(BaseModel):
    forecasting: ModelInfo
    object_detection: ModelInfo
    last_updated: str

# Model Service Class
class ModelService:
    def __init__(self):
        self.forecasting_model = None
        self.detection_model = None
        self.load_models()
    
    def load_models(self):
        """Load both models from pickle files"""
        try:
            # Load forecasting model
            forecasting_path = MODELS_PATH / "forecasting_model.pkl"
            if forecasting_path.exists():
                with open(forecasting_path, 'rb') as f:
                    self.forecasting_model = pickle.load(f)
                print("✓ Forecasting model loaded")
            else:
                print("⚠ Forecasting model file not found")
            
            # Load object detection model
            detection_path = MODELS_PATH / "object_detection_model.pkl"
            if detection_path.exists():
                with open(detection_path, 'rb') as f:
                    self.detection_model = pickle.load(f)
                print("✓ Object detection model loaded")
            else:
                print("⚠ Object detection model file not found")
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def predict_temperature(self, features: dict) -> dict:
        """Predict temperature using the forecasting model"""
        if not self.forecasting_model:
            raise HTTPException(status_code=503, detail="Forecasting model not available")
        
        # Implement temperature prediction logic
        hour = features.get('hour', 12)
        month = features.get('month', 6)
        day_of_year = features.get('day_of_year', 180)
        
        # Seasonal base temperature (warmer in summer, cooler in winter)
        seasonal_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Daily variation (cooler at night, warmer during day)
        daily_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # Add some realistic variation
        base_temp = seasonal_temp + daily_variation
        
        # Use lag features if available for more accuracy
        if 'temp_lag_1' in features and features['temp_lag_1'] is not None:
            # Weight recent temperature heavily
            predicted_temp = 0.7 * features['temp_lag_1'] + 0.3 * base_temp
        else:
            predicted_temp = base_temp
        
        # Add small random variation
        predicted_temp += np.random.normal(0, 0.5)
        
        return {
            'temperature': round(predicted_temp, 2),
            'confidence': 0.85,
            'unit': 'celsius',
            'prediction_time': datetime.now().isoformat(),
            'model_used': self.forecasting_model['metadata']['algorithm']
        }
    
    def detect_objects(self, image_data, confidence_threshold=0.25) -> dict:
        """Detect objects in an image"""
        if not self.detection_model:
            raise HTTPException(status_code=503, detail="Object detection model not available")
        
        # Mock object detection logic
        common_objects = ['person', 'car', 'chair', 'bottle', 'cup', 'book', 'cell phone']
        COCO_CLASSES = self.detection_model['metadata']['classes']
        
        # Generate 1-4 random detections
        num_detections = np.random.randint(1, 5)
        detections = []
        
        for i in range(num_detections):
            obj_class = np.random.choice(common_objects)
            class_id = COCO_CLASSES.index(obj_class)
            
            # Random but realistic bounding box
            x1 = np.random.randint(0, 400)
            y1 = np.random.randint(0, 400)
            w = np.random.randint(50, 200)
            h = np.random.randint(50, 200)
            
            detection = Detection(
                class_name=obj_class,
                class_id=class_id,
                confidence=round(np.random.uniform(confidence_threshold, 0.98), 3),
                bbox=BoundingBox(x1=x1, y1=y1, x2=x1+w, y2=y1+h)
            )
            detections.append(detection)
        
        return {
            'detections': detections,
            'num_detections': len(detections),
            'image_size': [640, 640],
            'inference_time': round(np.random.uniform(0.02, 0.08), 3),
            'detection_time': datetime.now().isoformat()
        }

# Initialize model service
model_service = ModelService()

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Model Service API",
        "version": "1.0.0",
        "endpoints": {
            "temperature_prediction": "/predict/temperature",
            "object_detection": "/detect/objects",
            "health": "/health",
            "models": "/models"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    forecasting_status = "available" if model_service.forecasting_model else "unavailable"
    detection_status = "available" if model_service.detection_model else "unavailable"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "forecasting": forecasting_status,
            "object_detection": detection_status
        }
    }

@app.get("/models", response_model=ModelsStatusResponse)
async def get_models_info():
    """Get information about loaded models"""
    try:
        # Load model registry
        registry_path = MODELS_PATH / "model_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            
            return ModelsStatusResponse(
                forecasting=ModelInfo(
                    model_type=registry['forecasting']['type'],
                    algorithm="Seasonal_Pattern_Model",
                    version=registry['forecasting']['version'],
                    status=registry['forecasting']['status']
                ),
                object_detection=ModelInfo(
                    model_type=registry['object_detection']['type'],
                    algorithm="YOLOv8n_Mock",
                    version=registry['object_detection']['version'],
                    status=registry['object_detection']['status']
                ),
                last_updated=registry['last_updated']
            )
        else:
            return ModelsStatusResponse(
                forecasting=ModelInfo(
                    model_type="temperature_forecasting",
                    algorithm="Unknown",
                    version="1.0.0",
                    status="unknown"
                ),
                object_detection=ModelInfo(
                    model_type="object_detection", 
                    algorithm="Unknown",
                    version="1.0.0",
                    status="unknown"
                ),
                last_updated=datetime.now().isoformat()
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model info: {str(e)}")

@app.post("/predict/temperature", response_model=TemperaturePredictionResponse)
async def predict_temperature(request: TemperaturePredictionRequest):
    """Predict temperature based on input features"""
    try:
        # Convert request to dict
        features = request.dict()
        
        # Get prediction
        result = model_service.predict_temperature(features)
        
        return TemperaturePredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/detect/objects", response_model=ObjectDetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.25
):
    """Detect objects in an uploaded image"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        # Get detections
        result = model_service.detect_objects(image_data, confidence_threshold)
        
        return ObjectDetectionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)