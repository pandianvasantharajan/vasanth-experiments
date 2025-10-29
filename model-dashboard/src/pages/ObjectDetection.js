import React, { useState, useCallback } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  Alert,
  Paper,
  CircularProgress,
  Chip,
  Slider,
  List,
  ListItem,
  ListItemText,
  Divider,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  CameraAlt as CameraIcon,
  CloudUpload as UploadIcon,
  Delete as DeleteIcon,
  Visibility as VisibilityIcon,
  Info as InfoIcon,
  Download as DownloadIcon
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { detectObjects, validateImageFile } from '../services/api';

const ObjectDetection = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [detectionResults, setDetectionResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25);

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    if (rejectedFiles.length > 0) {
      setError('Please upload a valid image file (JPEG, PNG, WebP)');
      return;
    }

    const file = acceptedFiles[0];
    if (file) {
      const validationErrors = validateImageFile(file);
      if (validationErrors.length > 0) {
        setError(validationErrors.join(', '));
        return;
      }

      setSelectedFile(file);
      setError(null);
      setDetectionResults(null);

      // Create preview
      const reader = new FileReader();
      reader.onload = () => setPreview(reader.result);
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    maxFiles: 1,
    multiple: false
  });

  const handleDetection = async () => {
    if (!selectedFile) {
      setError('Please select an image file first');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const results = await detectObjects(selectedFile, confidenceThreshold);
      setDetectionResults(results);
    } catch (err) {
      setError(err.message);
      setDetectionResults(null);
    } finally {
      setLoading(false);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setPreview(null);
    setDetectionResults(null);
    setError(null);
  };

  const downloadResults = () => {
    if (!detectionResults) return;

    const dataStr = JSON.stringify(detectionResults, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `detection_results_${new Date().toISOString().slice(0,19).replace(/:/g, '-')}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.5) return 'warning';
    return 'error';
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 2, mb: 4 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <CameraIcon color="primary" sx={{ fontSize: 40 }} />
          Object Detection
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Upload an image to detect and classify objects using YOLO-based computer vision
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Image Upload and Preview */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <UploadIcon color="primary" />
                Image Upload
                <Tooltip title="Supports JPEG, PNG, and WebP formats up to 10MB">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Typography>

              {/* File Upload Area */}
              {!selectedFile && (
                <Paper
                  {...getRootProps()}
                  sx={{
                    p: 4,
                    textAlign: 'center',
                    border: '2px dashed',
                    borderColor: isDragActive ? 'primary.main' : 'grey.300',
                    bgcolor: isDragActive ? 'action.hover' : 'background.paper',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    '&:hover': {
                      borderColor: 'primary.main',
                      bgcolor: 'action.hover'
                    }
                  }}
                >
                  <input {...getInputProps()} />
                  <UploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h6" gutterBottom>
                    {isDragActive ? 'Drop the image here' : 'Drag & drop an image here'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    or click to select a file
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Supports JPEG, PNG, WebP (max 10MB)
                  </Typography>
                </Paper>
              )}

              {/* Image Preview */}
              {preview && (
                <Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="subtitle1" fontWeight="medium">
                      Selected Image: {selectedFile?.name}
                    </Typography>
                    <Button
                      variant="outlined"
                      color="error"
                      size="small"
                      startIcon={<DeleteIcon />}
                      onClick={handleRemoveFile}
                    >
                      Remove
                    </Button>
                  </Box>

                  <Paper sx={{ p: 2, textAlign: 'center' }}>
                    <img
                      src={preview}
                      alt="Preview"
                      style={{
                        maxWidth: '100%',
                        maxHeight: '400px',
                        objectFit: 'contain',
                        borderRadius: '8px'
                      }}
                    />
                  </Paper>

                  {/* Confidence Threshold Slider */}
                  <Box sx={{ mt: 3 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Confidence Threshold: {confidenceThreshold}
                    </Typography>
                    <Slider
                      value={confidenceThreshold}
                      onChange={(e, value) => setConfidenceThreshold(value)}
                      min={0.1}
                      max={1.0}
                      step={0.05}
                      marks={[
                        { value: 0.1, label: '0.1' },
                        { value: 0.25, label: '0.25' },
                        { value: 0.5, label: '0.5' },
                        { value: 0.75, label: '0.75' },
                        { value: 1.0, label: '1.0' }
                      ]}
                      sx={{ mt: 1 }}
                    />
                    <Typography variant="caption" color="text.secondary">
                      Lower values detect more objects but may include false positives
                    </Typography>
                  </Box>

                  {/* Detect Button */}
                  <Button
                    variant="contained"
                    size="large"
                    startIcon={loading ? <CircularProgress size={20} /> : <VisibilityIcon />}
                    onClick={handleDetection}
                    disabled={loading}
                    sx={{ mt: 3 }}
                    fullWidth
                  >
                    {loading ? 'Detecting Objects...' : 'Detect Objects'}
                  </Button>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Results and Information */}
        <Grid item xs={12} md={4}>
          {/* Detection Results */}
          {detectionResults && (
            <Card sx={{ mb: 2 }}>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" color="primary">
                    Detection Results
                  </Typography>
                  <IconButton onClick={downloadResults} size="small">
                    <DownloadIcon />
                  </IconButton>
                </Box>

                {/* Summary Stats */}
                <Paper sx={{ p: 2, mb: 2, bgcolor: 'primary.main', color: 'white', textAlign: 'center' }}>
                  <Typography variant="h4" fontWeight="bold">
                    {detectionResults.num_detections}
                  </Typography>
                  <Typography variant="body1">
                    Objects Detected
                  </Typography>
                </Paper>

                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Inference Time
                    </Typography>
                    <Typography variant="body1" fontWeight="medium">
                      {(detectionResults.inference_time * 1000).toFixed(1)}ms
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Image Size
                    </Typography>
                    <Typography variant="body1" fontWeight="medium">
                      {detectionResults.image_size.join('×')}
                    </Typography>
                  </Grid>
                </Grid>

                {/* Detected Objects List */}
                {detectionResults.detections.length > 0 && (
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Detected Objects:
                    </Typography>
                    <List dense>
                      {detectionResults.detections.map((detection, index) => (
                        <React.Fragment key={index}>
                          <ListItem sx={{ px: 0 }}>
                            <ListItemText
                              primary={
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                  <Typography variant="body2" fontWeight="medium">
                                    {detection.class_name}
                                  </Typography>
                                  <Chip
                                    label={`${(detection.confidence * 100).toFixed(1)}%`}
                                    size="small"
                                    color={getConfidenceColor(detection.confidence)}
                                  />
                                </Box>
                              }
                              secondary={
                                <Typography variant="caption" color="text.secondary">
                                  Bbox: [{detection.bbox.x1.toFixed(0)}, {detection.bbox.y1.toFixed(0)}, {detection.bbox.x2.toFixed(0)}, {detection.bbox.y2.toFixed(0)}]
                                </Typography>
                              }
                            />
                          </ListItem>
                          {index < detectionResults.detections.length - 1 && <Divider />}
                        </React.Fragment>
                      ))}
                    </List>
                  </Box>
                )}

                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                  Detected at: {new Date(detectionResults.detection_time).toLocaleString()}
                </Typography>
              </CardContent>
            </Card>
          )}

          {/* Error Messages */}
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {/* Information Card */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                About Object Detection
              </Typography>

              <Typography variant="body2" paragraph>
                This model uses YOLO (You Only Look Once) architecture to detect and classify objects in images. 
                It can identify 80 different object classes from the COCO dataset.
              </Typography>

              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Supported Object Classes:
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {['person', 'car', 'bicycle', 'dog', 'cat', 'bird', 'chair', 'bottle', 'laptop', 'phone'].map((cls) => (
                    <Chip key={cls} label={cls} size="small" variant="outlined" />
                  ))}
                  <Chip label="... +70 more" size="small" variant="outlined" color="primary" />
                </Box>
              </Box>

              <Box sx={{ p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
                <Typography variant="body2" fontWeight="medium" gutterBottom>
                  💡 Tips for better results:
                </Typography>
                <Typography variant="body2" component="ul" sx={{ pl: 2, m: 0 }}>
                  <li>Use high-quality, well-lit images</li>
                  <li>Avoid heavily blurred or dark images</li>
                  <li>Lower confidence threshold for more detections</li>
                  <li>Higher confidence threshold for more accurate results</li>
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default ObjectDetection;