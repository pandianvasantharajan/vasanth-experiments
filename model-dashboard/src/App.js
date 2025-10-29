import React, { useState, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box, useTheme, useMediaQuery } from '@mui/material';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import TemperaturePrediction from './pages/TemperaturePrediction';
import ObjectDetection from './pages/ObjectDetection';
import { checkServiceHealth } from './services/api';

const DRAWER_WIDTH = 280;

function App() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [mobileOpen, setMobileOpen] = useState(false);
  const [serviceStatus, setServiceStatus] = useState({
    healthy: false,
    forecasting: 'unknown',
    objectDetection: 'unknown'
  });

  // Check service health on app load
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await checkServiceHealth();
        setServiceStatus(health);
      } catch (error) {
        console.error('Failed to check service health:', error);
        setServiceStatus({
          healthy: false,
          forecasting: 'unavailable',
          objectDetection: 'unavailable'
        });
      }
    };

    checkHealth();
    // Check health every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleDrawerClose = () => {
    setMobileOpen(false);
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      {/* Header */}
      <Header 
        drawerWidth={DRAWER_WIDTH}
        onMenuClick={handleDrawerToggle}
        serviceStatus={serviceStatus}
      />
      
      {/* Sidebar Navigation */}
      <Sidebar
        drawerWidth={DRAWER_WIDTH}
        mobileOpen={mobileOpen}
        onDrawerToggle={handleDrawerToggle}
        onDrawerClose={handleDrawerClose}
        isMobile={isMobile}
      />
      
      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: { xs: 1, sm: 2, md: 3 },
          width: { md: `calc(100% - ${DRAWER_WIDTH}px)` },
          mt: '64px', // Header height
          backgroundColor: 'background.default',
          minHeight: 'calc(100vh - 64px)',
        }}
      >
        <Routes>
          <Route path="/" element={<Dashboard serviceStatus={serviceStatus} />} />
          <Route path="/dashboard" element={<Dashboard serviceStatus={serviceStatus} />} />
          <Route path="/temperature" element={<TemperaturePrediction />} />
          <Route path="/object-detection" element={<ObjectDetection />} />
        </Routes>
      </Box>
    </Box>
  );
}

export default App;