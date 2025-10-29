# Simple Model Export Script
# Create example pickle files for both models for FastAPI demonstration

import pickle
import numpy as np
from datetime import datetime
import os

def create_forecasting_model():
    """Create a simple forecasting model for demonstration"""
    
    # Simple temperature prediction function
    def predict_temperature(features):
        """
        Predict temperature based on simple seasonal patterns
        
        Args:
            features (dict): Input features containing at minimum:
                - hour: int (0-23)
                - month: int (1-12)
                - day_of_year: int (1-365)
        
        Returns:
            dict: Prediction result
        """
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
        if 'temp_lag_1' in features:
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
            'prediction_time': datetime.now().isoformat()
        }
    
    # Model metadata
    model_info = {
        'model_type': 'forecasting',
        'algorithm': 'Seasonal_Pattern_Model',
        'target': 'temperature',
        'input_features': [
            'hour', 'month', 'day_of_year', 'day_of_week',
            'temp_lag_1', 'temp_lag_6', 'temp_lag_24'
        ],
        'performance_metrics': {
            'rmse': 2.1,
            'mae': 1.6,
            'r2': 0.88
        },
        'training_date': datetime.now().isoformat(),
        'version': '1.0.0'
    }
    
    return {
        'predict_function': predict_temperature,
        'metadata': model_info,
        'export_date': datetime.now().isoformat()
    }

def create_object_detection_model():
    """Create a simple object detection model for demonstration"""
    
    # COCO class names
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    def detect_objects(image_data, confidence_threshold=0.25):
        """
        Mock object detection function
        
        Args:
            image_data: Image file path or image data
            confidence_threshold: Minimum confidence for detections
        
        Returns:
            dict: Detection results
        """
        # Mock realistic detections based on common objects
        common_objects = ['person', 'car', 'chair', 'bottle', 'cup', 'book', 'cell phone']
        
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
            
            detection = {
                'class': obj_class,
                'class_id': class_id,
                'confidence': round(np.random.uniform(confidence_threshold, 0.98), 3),
                'bbox': [x1, y1, x1 + w, y1 + h]  # x1, y1, x2, y2
            }
            detections.append(detection)
        
        return {
            'detections': detections,
            'num_detections': len(detections),
            'image_size': [640, 640],
            'inference_time': round(np.random.uniform(0.02, 0.08), 3),
            'detection_time': datetime.now().isoformat()
        }
    
    # Model metadata
    model_info = {
        'model_type': 'object_detection',
        'algorithm': 'YOLOv8n_Mock',
        'dataset': 'COCO 2017',
        'num_classes': 80,
        'classes': COCO_CLASSES,
        'input_size': [640, 640],
        'performance_metrics': {
            'map50': 0.37,
            'map50_95': 0.28,
            'precision': 0.85,
            'recall': 0.75
        },
        'training_date': datetime.now().isoformat(),
        'version': '1.0.0'
    }
    
    return {
        'detect_function': detect_objects,
        'metadata': model_info,
        'export_date': datetime.now().isoformat()
    }

def main():
    """Export both models to pickle files"""
    print("=== CREATING MODEL PICKLE FILES ===")
    
    # Create forecasting model data (without functions)
    forecasting_data = {
        'metadata': {
            'model_type': 'forecasting',
            'algorithm': 'Seasonal_Pattern_Model',
            'target': 'temperature',
            'input_features': [
                'hour', 'month', 'day_of_year', 'day_of_week',
                'temp_lag_1', 'temp_lag_6', 'temp_lag_24'
            ],
            'performance_metrics': {
                'rmse': 2.1,
                'mae': 1.6,
                'r2': 0.88
            },
            'training_date': datetime.now().isoformat(),
            'version': '1.0.0'
        },
        'export_date': datetime.now().isoformat()
    }
    
    forecasting_path = "forecasting_model.pkl"
    
    with open(forecasting_path, 'wb') as f:
        pickle.dump(forecasting_data, f)
    
    print(f"✓ Forecasting model saved to: {forecasting_path}")
    print(f"  Algorithm: {forecasting_data['metadata']['algorithm']}")
    print(f"  RMSE: {forecasting_data['metadata']['performance_metrics']['rmse']}")
    
    # Create object detection model data (without functions)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    detection_data = {
        'metadata': {
            'model_type': 'object_detection',
            'algorithm': 'YOLOv8n_Mock',
            'dataset': 'COCO 2017',
            'num_classes': 80,
            'classes': COCO_CLASSES,
            'input_size': [640, 640],
            'performance_metrics': {
                'map50': 0.37,
                'map50_95': 0.28,
                'precision': 0.85,
                'recall': 0.75
            },
            'training_date': datetime.now().isoformat(),
            'version': '1.0.0'
        },
        'export_date': datetime.now().isoformat()
    }
    
    detection_path = "object_detection_model.pkl"
    
    with open(detection_path, 'wb') as f:
        pickle.dump(detection_data, f)
    
    print(f"✓ Object detection model saved to: {detection_path}")
    print(f"  Algorithm: {detection_data['metadata']['algorithm']}")
    print(f"  Classes: {detection_data['metadata']['num_classes']}")
    
    # Create a combined model info file
    model_registry = {
        'forecasting': {
            'file': forecasting_path,
            'type': 'temperature_forecasting',
            'status': 'active',
            'version': forecasting_data['metadata']['version']
        },
        'object_detection': {
            'file': detection_path,
            'type': 'object_detection',
            'status': 'active',
            'version': detection_data['metadata']['version']
        },
        'last_updated': datetime.now().isoformat()
    }
    
    with open('model_registry.json', 'w') as f:
        import json
        json.dump(model_registry, f, indent=2)
    
    print(f"✓ Model registry created: model_registry.json")
    print("\nAll models exported successfully!")

if __name__ == "__main__":
    main()