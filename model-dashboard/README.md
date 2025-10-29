# AI Model Services Dashboard

A comprehensive React dashboard for interacting with AI model services including temperature forecasting and object detection. Built with Material-UI and featuring a responsive design with collapsible sidebar navigation.

## Features

### 🌡️ Temperature Prediction
- **Temporal Feature Input**: Hour, month, day of year, day of week
- **Historical Data Support**: Optional lag temperature values for improved accuracy
- **Quick Examples**: Pre-configured seasonal examples (summer, winter, spring)
- **Real-time Validation**: Input validation with helpful error messages
- **Detailed Results**: Temperature prediction with confidence scores

### 📷 Object Detection
- **Drag & Drop Upload**: Easy image upload with preview
- **Confidence Threshold Control**: Adjustable detection sensitivity
- **COCO Dataset Support**: 80 different object classes
- **Bounding Box Visualization**: Detailed detection results with coordinates
- **Export Results**: Download detection results as JSON

### 🎛️ Dashboard Features
- **Service Status Monitoring**: Real-time health checks for both models
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Collapsible Sidebar**: Clean navigation with model information
- **Material-UI Design**: Modern, accessible user interface
- **Error Handling**: Comprehensive error messages and validation

## Technology Stack

- **Frontend**: React 18, Material-UI 5
- **Routing**: React Router DOM
- **HTTP Client**: Axios
- **File Upload**: React Dropzone
- **Icons**: Material-UI Icons
- **Build Tool**: Create React App

## Prerequisites

- Node.js 16+ and npm/yarn
- FastAPI model service running on port 8002
- Modern web browser with JavaScript enabled

## Installation

1. **Navigate to the project directory**:
   ```bash
   cd model-dashboard
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Set up environment variables** (optional):
   Create a `.env` file in the root directory:
   ```env
   REACT_APP_API_URL=http://localhost:8002
   ```

4. **Start the development server**:
   ```bash
   npm start
   ```

5. **Open your browser**:
   Navigate to `http://localhost:3000`

## Project Structure

```
model-dashboard/
├── public/
│   ├── index.html
│   └── manifest.json
├── src/
│   ├── components/
│   │   ├── Header.js          # Top navigation bar
│   │   └── Sidebar.js         # Collapsible sidebar navigation
│   ├── pages/
│   │   ├── Dashboard.js       # Main dashboard with service overview
│   │   ├── TemperaturePrediction.js  # Temperature forecasting interface
│   │   └── ObjectDetection.js # Object detection interface
│   ├── services/
│   │   └── api.js            # API client and service functions
│   ├── App.js                # Main application component
│   └── index.js              # Application entry point
├── package.json
└── README.md
```

## API Integration

The dashboard connects to your FastAPI model service with the following endpoints:

### Service Health
- `GET /health` - Check service and model availability
- `GET /models` - Get model metadata and information
- `GET /` - Get API information

### Temperature Prediction
- `POST /predict/temperature` - Predict temperature
  ```json
  {
    "hour": 14,
    "month": 7,
    "day_of_year": 195,
    "day_of_week": 3,
    "temp_lag_1": 25.5,
    "temp_lag_6": 23.8,
    "temp_lag_24": 22.1
  }
  ```

### Object Detection
- `POST /detect/objects` - Detect objects in uploaded image
  - Multipart form data with image file
  - Optional confidence threshold parameter

## Usage Guide

### Temperature Prediction

1. **Navigate to Temperature Prediction** page from the sidebar
2. **Enter temporal features**:
   - Hour (0-23): Current hour in 24-hour format
   - Month (1-12): Current month
   - Day of year (1-365): Day number in the year
   - Day of week (0-6): Optional, Monday=0 to Sunday=6

3. **Add historical data** (optional but recommended):
   - Temperature 1 hour ago
   - Temperature 6 hours ago
   - Temperature 24 hours ago

4. **Use quick examples** for testing:
   - Summer Day: Hot weather scenario
   - Winter Day: Cold weather scenario
   - Spring Day: Mild weather scenario

5. **Click "Predict Temperature"** to get results
6. **View results** with confidence score and model information

### Object Detection

1. **Navigate to Object Detection** page from the sidebar
2. **Upload an image**:
   - Drag and drop an image file
   - Or click to select from file browser
   - Supported formats: JPEG, PNG, WebP (max 10MB)

3. **Adjust confidence threshold** (0.1 to 1.0):
   - Lower values: More detections, possible false positives
   - Higher values: Fewer but more accurate detections

4. **Click "Detect Objects"** to analyze the image
5. **View results**:
   - Number of objects detected
   - List of detected objects with confidence scores
   - Bounding box coordinates
   - Inference time

6. **Download results** as JSON file (optional)

### Dashboard Overview

- **Service Status**: Monitor the health of both AI models
- **Quick Actions**: Direct links to prediction interfaces
- **Model Information**: View model metadata and versions
- **API Information**: Check available endpoints and versions

## Customization

### Styling and Theme

The app uses Material-UI's theming system. To customize:

1. **Edit the theme** in `src/index.js`:
   ```javascript
   const theme = createTheme({
     palette: {
       primary: { main: '#1976d2' },
       secondary: { main: '#dc004e' },
     },
   });
   ```

2. **Modify component styles** using the `sx` prop or styled components

### API Configuration

1. **Change API base URL** in `src/services/api.js`:
   ```javascript
   const API_BASE_URL = 'http://your-api-server:port';
   ```

2. **Modify request timeout** or headers in the axios configuration

### Adding New Features

1. **Create new page components** in `src/pages/`
2. **Add routes** in `src/App.js`
3. **Update sidebar navigation** in `src/components/Sidebar.js`
4. **Add API functions** in `src/services/api.js`

## Available Scripts

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run test suite
- `npm eject` - Eject from Create React App (⚠️ irreversible)

## Deployment

### Development Deployment
```bash
npm start
```

### Production Build
```bash
npm run build
```

### Deploy to Static Hosting
1. Build the app: `npm run build`
2. Upload the `build/` folder to your hosting service
3. Configure routing for single-page application
4. Set environment variables for production API URL

### Docker Deployment
Create a `Dockerfile`:
```dockerfile
FROM node:16-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## Troubleshooting

### Common Issues

1. **API Connection Errors**:
   - Check if the FastAPI service is running on port 8002
   - Verify CORS is enabled on the backend
   - Check network connectivity

2. **Image Upload Issues**:
   - Ensure file size is under 10MB
   - Use supported formats (JPEG, PNG, WebP)
   - Check browser permissions for file access

3. **Build Errors**:
   - Clear node_modules and reinstall: `rm -rf node_modules && npm install`
   - Check Node.js version compatibility
   - Update dependencies if needed

### Performance Optimization

1. **Image Optimization**:
   - Compress images before upload
   - Use appropriate image formats
   - Consider image resizing for large files

2. **Network Optimization**:
   - Enable HTTP/2 on your server
   - Use CDN for static assets
   - Implement proper caching headers

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit with descriptive messages
5. Push to your branch and create a pull request

## License

This project is part of the vasanth-experiments monorepo and follows the same licensing terms.