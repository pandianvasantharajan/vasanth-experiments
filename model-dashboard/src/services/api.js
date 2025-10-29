import axios from 'axios';

// Base URL for the model service API
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8002';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for debugging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    
    if (error.response?.status === 404) {
      throw new Error('Service endpoint not found. Please check if the model service is running.');
    } else if (error.response?.status === 500) {
      throw new Error('Internal server error. Please try again later.');
    } else if (error.code === 'ECONNREFUSED') {
      throw new Error('Cannot connect to model service. Please ensure the service is running on port 8002.');
    }
    
    throw error;
  }
);

// API Functions

/**
 * Check service health and model availability
 */
export const checkServiceHealth = async () => {
  try {
    const response = await apiClient.get('/health');
    return {
      healthy: response.data.status === 'healthy',
      forecasting: response.data.models?.forecasting || 'unknown',
      objectDetection: response.data.models?.object_detection || 'unknown',
      timestamp: response.data.timestamp
    };
  } catch (error) {
    throw new Error(`Health check failed: ${error.message}`);
  }
};

/**
 * Get model information and metadata
 */
export const getModelsInfo = async () => {
  try {
    const response = await apiClient.get('/models');
    return response.data;
  } catch (error) {
    throw new Error(`Failed to get models info: ${error.message}`);
  }
};

/**
 * Predict temperature based on input features
 */
export const predictTemperature = async (features) => {
  try {
    const response = await apiClient.post('/predict/temperature', features);
    return response.data;
  } catch (error) {
    if (error.response?.status === 422) {
      throw new Error('Invalid input data. Please check your input values.');
    }
    throw new Error(`Temperature prediction failed: ${error.message}`);
  }
};

/**
 * Detect objects in uploaded image
 */
export const detectObjects = async (imageFile, confidenceThreshold = 0.25) => {
  try {
    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('confidence_threshold', confidenceThreshold.toString());

    const response = await apiClient.post('/detect/objects', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  } catch (error) {
    if (error.response?.status === 400) {
      throw new Error('Invalid image file. Please upload a valid image (JPEG, PNG).');
    } else if (error.response?.status === 422) {
      throw new Error('Invalid request data. Please check your inputs.');
    }
    throw new Error(`Object detection failed: ${error.message}`);
  }
};

/**
 * Get service root information
 */
export const getServiceInfo = async () => {
  try {
    const response = await apiClient.get('/');
    return response.data;
  } catch (error) {
    throw new Error(`Failed to get service info: ${error.message}`);
  }
};

// Utility functions

/**
 * Validate temperature prediction input
 */
export const validateTemperatureInput = (data) => {
  const errors = [];
  
  if (data.hour < 0 || data.hour > 23) {
    errors.push('Hour must be between 0 and 23');
  }
  
  if (data.month < 1 || data.month > 12) {
    errors.push('Month must be between 1 and 12');
  }
  
  if (data.day_of_year < 1 || data.day_of_year > 365) {
    errors.push('Day of year must be between 1 and 365');
  }
  
  if (data.day_of_week !== undefined && (data.day_of_week < 0 || data.day_of_week > 6)) {
    errors.push('Day of week must be between 0 and 6');
  }
  
  return errors;
};

/**
 * Validate image file for object detection
 */
export const validateImageFile = (file) => {
  const errors = [];
  
  if (!file) {
    errors.push('Please select an image file');
    return errors;
  }
  
  // Check file type
  const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
  if (!validTypes.includes(file.type)) {
    errors.push('Please upload a valid image file (JPEG, PNG, WebP)');
  }
  
  // Check file size (max 10MB)
  const maxSize = 10 * 1024 * 1024; // 10MB
  if (file.size > maxSize) {
    errors.push('Image file size must be less than 10MB');
  }
  
  return errors;
};

export default apiClient;