import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  Chip,
  Tooltip
} from '@mui/material';
import {
  Menu as MenuIcon,
  Circle as CircleIcon
} from '@mui/icons-material';

const Header = ({ drawerWidth, onMenuClick, serviceStatus }) => {
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

  const getStatusText = (status) => {
    switch (status) {
      case 'available':
        return 'Online';
      case 'unavailable':
        return 'Offline';
      default:
        return 'Unknown';
    }
  };

  return (
    <AppBar
      position="fixed"
      sx={{
        width: { md: `calc(100% - ${drawerWidth}px)` },
        ml: { md: `${drawerWidth}px` },
        zIndex: (theme) => theme.zIndex.drawer + 1,
      }}
    >
      <Toolbar>
        <IconButton
          color="inherit"
          aria-label="open drawer"
          edge="start"
          onClick={onMenuClick}
          sx={{ mr: 2, display: { md: 'none' } }}
        >
          <MenuIcon />
        </IconButton>
        
        <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
          AI Model Services Dashboard
        </Typography>
        
        {/* Service Status Indicators */}
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Tooltip title={`Temperature Forecasting Service: ${getStatusText(serviceStatus.forecasting)}`}>
            <Chip
              icon={<CircleIcon sx={{ fontSize: 12 }} />}
              label="Forecasting"
              size="small"
              color={getStatusColor(serviceStatus.forecasting)}
              variant="outlined"
              sx={{ 
                color: 'white',
                borderColor: 'rgba(255, 255, 255, 0.3)',
                '& .MuiChip-icon': { color: 'inherit' }
              }}
            />
          </Tooltip>
          
          <Tooltip title={`Object Detection Service: ${getStatusText(serviceStatus.objectDetection)}`}>
            <Chip
              icon={<CircleIcon sx={{ fontSize: 12 }} />}
              label="Detection"
              size="small"
              color={getStatusColor(serviceStatus.objectDetection)}
              variant="outlined"
              sx={{ 
                color: 'white',
                borderColor: 'rgba(255, 255, 255, 0.3)',
                '& .MuiChip-icon': { color: 'inherit' }
              }}
            />
          </Tooltip>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;