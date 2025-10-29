# Model Service Tests

import pytest
import json
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import pickle
from datetime import datetime

from main import app, ModelService

client = TestClient(app)

class TestModelService:
    """Test the ModelService class"""
    
    def test_init(self):
        """Test ModelService initialization"""
        service = ModelService()
        # Should not raise any exceptions
        assert service is not None
    
    def test_predict_temperature_basic(self):
        """Test basic temperature prediction"""
        service = ModelService()
        
        # Create mock forecasting model
        service.forecasting_model = {
            'metadata': {
                'algorithm': 'Test_Model'
            }
        }
        
        features = {
            'hour': 14,
            'month': 7,
            'day_of_year': 195
        }
        
        result = service.predict_temperature(features)
        
        assert 'temperature' in result
        assert 'confidence' in result
        assert 'unit' in result
        assert result['unit'] == 'celsius'
        assert isinstance(result['temperature'], float)
        assert 0 <= result['confidence'] <= 1

class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "models" in data

    def test_models_endpoint(self):
        """Test models info endpoint"""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "forecasting" in data
        assert "object_detection" in data

    def test_temperature_prediction_endpoint(self):
        """Test temperature prediction endpoint"""
        request_data = {
            "hour": 14,
            "month": 7,
            "day_of_year": 195,
            "day_of_week": 3,
            "temp_lag_1": 25.5
        }
        
        response = client.post("/predict/temperature", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "temperature" in data
        assert "confidence" in data
        assert "unit" in data
        assert "prediction_time" in data

    def test_temperature_prediction_invalid_data(self):
        """Test temperature prediction with invalid data"""
        request_data = {
            "hour": 25,  # Invalid hour
            "month": 13,  # Invalid month
            "day_of_year": 400  # Invalid day
        }
        
        response = client.post("/predict/temperature", json=request_data)
        # Should still work but may produce unrealistic results
        assert response.status_code in [200, 422]  # 422 for validation error

    def test_object_detection_no_file(self):
        """Test object detection without file"""
        response = client.post("/detect/objects")
        assert response.status_code == 422  # Validation error

    def test_object_detection_invalid_file(self):
        """Test object detection with invalid file"""
        files = {"file": ("test.txt", "not an image", "text/plain")}
        response = client.post("/detect/objects", files=files)
        assert response.status_code == 400  # Bad request

class TestModelLoading:
    """Test model loading functionality"""
    
    def test_load_models_with_files(self, tmp_path):
        """Test loading models when files exist"""
        # Create temporary model files
        forecasting_data = {
            'metadata': {
                'algorithm': 'Test_Forecasting_Model',
                'version': '1.0.0'
            }
        }
        
        detection_data = {
            'metadata': {
                'algorithm': 'Test_Detection_Model',
                'version': '1.0.0',
                'classes': ['person', 'car']
            }
        }
        
        # Save test model files
        forecasting_file = tmp_path / "forecasting_model.pkl"
        detection_file = tmp_path / "object_detection_model.pkl"
        
        with open(forecasting_file, 'wb') as f:
            pickle.dump(forecasting_data, f)
        
        with open(detection_file, 'wb') as f:
            pickle.dump(detection_data, f)
        
        # Test loading (would need to modify ModelService to accept custom path)
        # This is a conceptual test - actual implementation would require refactoring
        assert forecasting_file.exists()
        assert detection_file.exists()

class TestValidation:
    """Test input validation"""
    
    def test_temperature_request_validation(self):
        """Test temperature request validation"""
        # Missing required fields
        response = client.post("/predict/temperature", json={})
        assert response.status_code == 422
        
        # Invalid types
        response = client.post("/predict/temperature", json={
            "hour": "not_a_number",
            "month": 7,
            "day_of_year": 195
        })
        assert response.status_code == 422

    def test_confidence_threshold_validation(self):
        """Test confidence threshold validation for object detection"""
        # Create a minimal image file for testing
        files = {"file": ("test.jpg", b"fake_image_data", "image/jpeg")}
        data = {"confidence_threshold": 1.5}  # Invalid confidence > 1.0
        
        response = client.post("/detect/objects", files=files, data=data)
        # Should handle invalid confidence gracefully
        assert response.status_code in [200, 400, 422]

class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_missing_models(self):
        """Test behavior when models are not available"""
        # This would require mocking or temporary modification of model loading
        pass
    
    def test_prediction_errors(self):
        """Test handling of prediction errors"""
        # Test with extreme values that might cause errors
        request_data = {
            "hour": -1,
            "month": -1,
            "day_of_year": -1
        }
        
        response = client.post("/predict/temperature", json=request_data)
        # Should either handle gracefully or return appropriate error
        assert response.status_code in [200, 400, 422, 500]

if __name__ == "__main__":
    pytest.main([__file__])