import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
  Alert,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Thermostat as ThermostatIcon,
  CameraAlt as CameraIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  Timeline as TimelineIcon,
  Psychology as PsychologyIcon
} from '@mui/icons-material';
import { getModelsInfo, getServiceInfo } from '../services/api';

const Dashboard = ({ serviceStatus }) => {
  const [modelsInfo, setModelsInfo] = useState(null);
  const [serviceInfo, setServiceInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [models, service] = await Promise.all([
          getModelsInfo(),
          getServiceInfo()
        ]);
        setModelsInfo(models);
        setServiceInfo(service);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'available':
        return <CheckCircleIcon color="success" />;
      case 'unavailable':
        return <ErrorIcon color="error" />;
      default:
        return <WarningIcon color="warning" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'available':
        return 'success';
      case 'unavailable':
        return 'error';
      default:
        return 'warning';
    }
  };

  if (loading) {
    return (
      <Container maxWidth="lg">
        <Box sx={{ mt: 4 }}>
          <Typography variant="h4" gutterBottom>
            Dashboard
          </Typography>
          <LinearProgress />
          <Typography sx={{ mt: 2 }}>Loading service information...</Typography>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 2, mb: 4 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <PsychologyIcon color="primary" sx={{ fontSize: 40 }} />
          AI Model Services Dashboard
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" gutterBottom>
          Monitor and interact with temperature forecasting and object detection models
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Service Status Overview */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <SpeedIcon color="primary" />
                Service Status
              </Typography>
              
              <List>
                <ListItem>
                  <ListItemIcon>
                    {getStatusIcon(serviceStatus.forecasting)}
                  </ListItemIcon>
                  <ListItemText
                    primary="Temperature Forecasting"
                    secondary="Time series prediction model"
                  />
                  <Chip 
                    label={serviceStatus.forecasting} 
                    color={getStatusColor(serviceStatus.forecasting)}
                    size="small"
                  />
                </ListItem>
                
                <Divider variant="inset" component="li" />
                
                <ListItem>
                  <ListItemIcon>
                    {getStatusIcon(serviceStatus.objectDetection)}
                  </ListItemIcon>
                  <ListItemText
                    primary="Object Detection"
                    secondary="Computer vision model"
                  />
                  <Chip 
                    label={serviceStatus.objectDetection} 
                    color={getStatusColor(serviceStatus.objectDetection)}
                    size="small"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <TimelineIcon color="primary" />
                Quick Actions
              </Typography>
              
              <List>
                <ListItem button component="a" href="/temperature">
                  <ListItemIcon>
                    <ThermostatIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Temperature Prediction"
                    secondary="Forecast temperature based on temporal features"
                  />
                </ListItem>
                
                <Divider variant="inset" component="li" />
                
                <ListItem button component="a" href="/object-detection">
                  <ListItemIcon>
                    <CameraIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Object Detection"
                    secondary="Detect and classify objects in images"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Temperature Forecasting Model Info */}
        {modelsInfo?.forecasting && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <ThermostatIcon color="primary" />
                  Temperature Forecasting Model
                </Typography>
                
                <Paper variant="outlined" sx={{ p: 2, mt: 2 }}>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Algorithm
                      </Typography>
                      <Typography variant="body1" fontWeight="medium">
                        {modelsInfo.forecasting.algorithm}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Version
                      </Typography>
                      <Typography variant="body1" fontWeight="medium">
                        {modelsInfo.forecasting.version}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Status
                      </Typography>
                      <Chip 
                        label={modelsInfo.forecasting.status} 
                        color="success" 
                        size="small" 
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Type
                      </Typography>
                      <Typography variant="body1" fontWeight="medium">
                        {modelsInfo.forecasting.model_type}
                      </Typography>
                    </Grid>
                  </Grid>
                </Paper>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Object Detection Model Info */}
        {modelsInfo?.object_detection && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <CameraIcon color="primary" />
                  Object Detection Model
                </Typography>
                
                <Paper variant="outlined" sx={{ p: 2, mt: 2 }}>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Algorithm
                      </Typography>
                      <Typography variant="body1" fontWeight="medium">
                        {modelsInfo.object_detection.algorithm}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Version
                      </Typography>
                      <Typography variant="body1" fontWeight="medium">
                        {modelsInfo.object_detection.version}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Status
                      </Typography>
                      <Chip 
                        label={modelsInfo.object_detection.status} 
                        color="success" 
                        size="small" 
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Type
                      </Typography>
                      <Typography variant="body1" fontWeight="medium">
                        {modelsInfo.object_detection.model_type}
                      </Typography>
                    </Grid>
                  </Grid>
                </Paper>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* API Information */}
        {serviceInfo && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <MemoryIcon color="primary" />
                  API Information
                </Typography>
                
                <Grid container spacing={3}>
                  <Grid item xs={12} md={4}>
                    <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="primary" fontWeight="bold">
                        {serviceInfo.version}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        API Version
                      </Typography>
                    </Paper>
                  </Grid>
                  
                  <Grid item xs={12} md={8}>
                    <Typography variant="subtitle2" gutterBottom>
                      Available Endpoints:
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {serviceInfo.endpoints && Object.entries(serviceInfo.endpoints).map(([key, path]) => (
                        <Chip
                          key={key}
                          label={`${key}: ${path}`}
                          variant="outlined"
                          size="small"
                        />
                      ))}
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Usage Instructions */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Getting Started
              </Typography>
              
              <Typography variant="body1" paragraph>
                This dashboard provides access to two AI models:
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Box sx={{ p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                    <Typography variant="subtitle1" fontWeight="medium" gutterBottom>
                      🌡️ Temperature Prediction
                    </Typography>
                    <Typography variant="body2">
                      Use temporal features like hour, month, day of year, and historical temperature data 
                      to predict future temperatures. Ideal for weather forecasting and climate analysis.
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Box sx={{ p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                    <Typography variant="subtitle1" fontWeight="medium" gutterBottom>
                      📷 Object Detection
                    </Typography>
                    <Typography variant="body2">
                      Upload images to detect and classify objects using YOLO-based computer vision. 
                      Supports 80 different object classes from the COCO dataset.
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Dashboard;