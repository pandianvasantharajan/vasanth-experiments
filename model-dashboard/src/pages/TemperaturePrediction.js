import React, { useState } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Box,
  Alert,
  Paper,
  CircularProgress,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  Thermostat as ThermostatIcon,
  Send as SendIcon,
  Info as InfoIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { predictTemperature, validateTemperatureInput } from '../services/api';

const TemperaturePrediction = () => {
  const [formData, setFormData] = useState({
    hour: 12,
    month: 7,
    day_of_year: 195,
    day_of_week: 3,
    temp_lag_1: '',
    temp_lag_6: '',
    temp_lag_24: ''
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [validationErrors, setValidationErrors] = useState([]);

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
    
    // Clear errors when user starts typing
    if (error) setError(null);
    if (validationErrors.length > 0) setValidationErrors([]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Prepare data for API
    const apiData = {
      hour: parseInt(formData.hour),
      month: parseInt(formData.month),
      day_of_year: parseInt(formData.day_of_year),
      day_of_week: formData.day_of_week ? parseInt(formData.day_of_week) : undefined,
      temp_lag_1: formData.temp_lag_1 ? parseFloat(formData.temp_lag_1) : undefined,
      temp_lag_6: formData.temp_lag_6 ? parseFloat(formData.temp_lag_6) : undefined,
      temp_lag_24: formData.temp_lag_24 ? parseFloat(formData.temp_lag_24) : undefined
    };

    // Validate input
    const errors = validateTemperatureInput(apiData);
    if (errors.length > 0) {
      setValidationErrors(errors);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setValidationErrors([]);
      
      const result = await predictTemperature(apiData);
      setPrediction(result);
    } catch (err) {
      setError(err.message);
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData({
      hour: 12,
      month: 7,
      day_of_year: 195,
      day_of_week: 3,
      temp_lag_1: '',
      temp_lag_6: '',
      temp_lag_24: ''
    });
    setPrediction(null);
    setError(null);
    setValidationErrors([]);
  };

  const setQuickExample = (example) => {
    const examples = {
      summer: {
        hour: 14,
        month: 7,
        day_of_year: 195,
        day_of_week: 3,
        temp_lag_1: '25.5',
        temp_lag_6: '23.8',
        temp_lag_24: '22.1'
      },
      winter: {
        hour: 10,
        month: 1,
        day_of_year: 15,
        day_of_week: 1,
        temp_lag_1: '2.1',
        temp_lag_6: '1.5',
        temp_lag_24: '0.8'
      },
      spring: {
        hour: 16,
        month: 4,
        day_of_year: 105,
        day_of_week: 5,
        temp_lag_1: '18.2',
        temp_lag_6: '16.7',
        temp_lag_24: '15.3'
      }
    };
    
    setFormData(examples[example]);
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 2, mb: 4 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <ThermostatIcon color="primary" sx={{ fontSize: 40 }} />
          Temperature Prediction
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Predict temperature using temporal features and historical data
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Input Form */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <ThermostatIcon color="primary" />
                Input Features
                <Tooltip title="Enter temporal features and optional historical temperature data">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Typography>

              {/* Quick Examples */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Quick Examples:
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  <Button size="small" variant="outlined" onClick={() => setQuickExample('summer')}>
                    Summer Day
                  </Button>
                  <Button size="small" variant="outlined" onClick={() => setQuickExample('winter')}>
                    Winter Day
                  </Button>
                  <Button size="small" variant="outlined" onClick={() => setQuickExample('spring')}>
                    Spring Day
                  </Button>
                  <Button size="small" variant="outlined" onClick={handleReset} startIcon={<RefreshIcon />}>
                    Reset
                  </Button>
                </Box>
              </Box>

              <form onSubmit={handleSubmit}>
                <Grid container spacing={3}>
                  {/* Time Features */}
                  <Grid item xs={12}>
                    <Typography variant="subtitle1" gutterBottom fontWeight="medium">
                      Time Features (Required)
                    </Typography>
                  </Grid>

                  <Grid item xs={12} sm={6} md={4}>
                    <TextField
                      fullWidth
                      label="Hour (0-23)"
                      type="number"
                      value={formData.hour}
                      onChange={(e) => handleInputChange('hour', e.target.value)}
                      inputProps={{ min: 0, max: 23 }}
                      required
                      helperText="Hour of the day (24-hour format)"
                    />
                  </Grid>

                  <Grid item xs={12} sm={6} md={4}>
                    <FormControl fullWidth required>
                      <InputLabel>Month</InputLabel>
                      <Select
                        value={formData.month}
                        label="Month"
                        onChange={(e) => handleInputChange('month', e.target.value)}
                      >
                        {Array.from({ length: 12 }, (_, i) => (
                          <MenuItem key={i + 1} value={i + 1}>
                            {new Date(2000, i, 1).toLocaleString('default', { month: 'long' })} ({i + 1})
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Grid>

                  <Grid item xs={12} sm={6} md={4}>
                    <TextField
                      fullWidth
                      label="Day of Year (1-365)"
                      type="number"
                      value={formData.day_of_year}
                      onChange={(e) => handleInputChange('day_of_year', e.target.value)}
                      inputProps={{ min: 1, max: 365 }}
                      required
                      helperText="Day number in the year"
                    />
                  </Grid>

                  <Grid item xs={12} sm={6}>
                    <FormControl fullWidth>
                      <InputLabel>Day of Week (Optional)</InputLabel>
                      <Select
                        value={formData.day_of_week}
                        label="Day of Week (Optional)"
                        onChange={(e) => handleInputChange('day_of_week', e.target.value)}
                      >
                        <MenuItem value="">Not specified</MenuItem>
                        <MenuItem value={0}>Monday</MenuItem>
                        <MenuItem value={1}>Tuesday</MenuItem>
                        <MenuItem value={2}>Wednesday</MenuItem>
                        <MenuItem value={3}>Thursday</MenuItem>
                        <MenuItem value={4}>Friday</MenuItem>
                        <MenuItem value={5}>Saturday</MenuItem>
                        <MenuItem value={6}>Sunday</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>

                  {/* Historical Temperature Features */}
                  <Grid item xs={12}>
                    <Typography variant="subtitle1" gutterBottom fontWeight="medium" sx={{ mt: 2 }}>
                      Historical Temperature Data (Optional - improves accuracy)
                    </Typography>
                  </Grid>

                  <Grid item xs={12} sm={4}>
                    <TextField
                      fullWidth
                      label="Temperature 1 hour ago (°C)"
                      type="number"
                      value={formData.temp_lag_1}
                      onChange={(e) => handleInputChange('temp_lag_1', e.target.value)}
                      inputProps={{ step: 0.1 }}
                      helperText="Recent temperature reading"
                    />
                  </Grid>

                  <Grid item xs={12} sm={4}>
                    <TextField
                      fullWidth
                      label="Temperature 6 hours ago (°C)"
                      type="number"
                      value={formData.temp_lag_6}
                      onChange={(e) => handleInputChange('temp_lag_6', e.target.value)}
                      inputProps={{ step: 0.1 }}
                      helperText="Temperature 6 hours ago"
                    />
                  </Grid>

                  <Grid item xs={12} sm={4}>
                    <TextField
                      fullWidth
                      label="Temperature 24 hours ago (°C)"
                      type="number"
                      value={formData.temp_lag_24}
                      onChange={(e) => handleInputChange('temp_lag_24', e.target.value)}
                      inputProps={{ step: 0.1 }}
                      helperText="Temperature yesterday"
                    />
                  </Grid>

                  {/* Submit Button */}
                  <Grid item xs={12}>
                    <Button
                      type="submit"
                      variant="contained"
                      size="large"
                      startIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
                      disabled={loading}
                      sx={{ mt: 2 }}
                    >
                      {loading ? 'Predicting...' : 'Predict Temperature'}
                    </Button>
                  </Grid>
                </Grid>
              </form>
            </CardContent>
          </Card>
        </Grid>

        {/* Results and Information */}
        <Grid item xs={12} md={4}>
          {/* Prediction Results */}
          {prediction && (
            <Card sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom color="primary">
                  Prediction Result
                </Typography>
                
                <Paper sx={{ p: 3, textAlign: 'center', bgcolor: 'primary.main', color: 'white' }}>
                  <Typography variant="h3" fontWeight="bold">
                    {prediction.temperature}°C
                  </Typography>
                  <Typography variant="body1">
                    Predicted Temperature
                  </Typography>
                </Paper>

                <Box sx={{ mt: 2 }}>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Confidence
                      </Typography>
                      <Typography variant="h6">
                        {(prediction.confidence * 100).toFixed(1)}%
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Model Used
                      </Typography>
                      <Chip label={prediction.model_used} size="small" />
                    </Grid>
                  </Grid>

                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Prediction Time
                    </Typography>
                    <Typography variant="body2">
                      {new Date(prediction.prediction_time).toLocaleString()}
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          )}

          {/* Error Messages */}
          {(error || validationErrors.length > 0) && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error && <Typography>{error}</Typography>}
              {validationErrors.map((err, index) => (
                <Typography key={index}>• {err}</Typography>
              ))}
            </Alert>
          )}

          {/* Information Card */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                About Temperature Prediction
              </Typography>
              
              <Typography variant="body2" paragraph>
                This model uses seasonal patterns and temporal features to predict temperature. 
                The prediction considers:
              </Typography>

              <Box component="ul" sx={{ mt: 1, pl: 2 }}>
                <Typography component="li" variant="body2">
                  <strong>Seasonal cycles:</strong> Natural temperature variations throughout the year
                </Typography>
                <Typography component="li" variant="body2">
                  <strong>Daily patterns:</strong> Temperature changes during the day
                </Typography>
                <Typography component="li" variant="body2">
                  <strong>Historical data:</strong> Recent temperature readings for better accuracy
                </Typography>
              </Box>

              <Box sx={{ mt: 2, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
                <Typography variant="body2" fontWeight="medium">
                  💡 Tip: Including historical temperature data significantly improves prediction accuracy!
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default TemperaturePrediction;