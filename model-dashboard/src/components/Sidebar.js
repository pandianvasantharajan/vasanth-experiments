import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Drawer,
  Box,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Avatar,
  useTheme
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Thermostat as ThermostatIcon,
  CameraAlt as CameraIcon,
  Psychology as PsychologyIcon
} from '@mui/icons-material';

const menuItems = [
  {
    text: 'Dashboard',
    icon: <DashboardIcon />,
    path: '/dashboard',
    description: 'Overview and service status'
  },
  {
    text: 'Temperature Prediction',
    icon: <ThermostatIcon />,
    path: '/temperature',
    description: 'Forecast temperature using time series data'
  },
  {
    text: 'Object Detection',
    icon: <CameraIcon />,
    path: '/object-detection',
    description: 'Detect objects in uploaded images'
  }
];

const Sidebar = ({ 
  drawerWidth, 
  mobileOpen, 
  onDrawerToggle, 
  onDrawerClose, 
  isMobile 
}) => {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();

  const handleItemClick = (path) => {
    navigate(path);
    if (isMobile) {
      onDrawerClose();
    }
  };

  const drawerContent = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header Section */}
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Avatar
          sx={{
            width: 56,
            height: 56,
            mx: 'auto',
            mb: 2,
            bgcolor: 'primary.main'
          }}
        >
          <PsychologyIcon sx={{ fontSize: 32 }} />
        </Avatar>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
          AI Model Services
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Temperature & Object Detection
        </Typography>
      </Box>

      <Divider />

      {/* Navigation Menu */}
      <Box sx={{ flexGrow: 1, pt: 2 }}>
        <List>
          {menuItems.map((item) => {
            const isActive = location.pathname === item.path || 
                           (location.pathname === '/' && item.path === '/dashboard');
            
            return (
              <ListItem key={item.text} disablePadding sx={{ px: 2, mb: 1 }}>
                <ListItemButton
                  onClick={() => handleItemClick(item.path)}
                  selected={isActive}
                  sx={{
                    borderRadius: 2,
                    mb: 0.5,
                    '&.Mui-selected': {
                      backgroundColor: theme.palette.primary.main,
                      color: 'white',
                      '&:hover': {
                        backgroundColor: theme.palette.primary.dark,
                      },
                      '& .MuiListItemIcon-root': {
                        color: 'white',
                      },
                    },
                    '&:hover': {
                      backgroundColor: isActive 
                        ? theme.palette.primary.dark 
                        : theme.palette.action.hover,
                    },
                  }}
                >
                  <ListItemIcon
                    sx={{
                      color: isActive ? 'white' : 'inherit',
                      minWidth: 40,
                    }}
                  >
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText
                    primary={item.text}
                    secondary={!isActive ? item.description : ''}
                    primaryTypographyProps={{
                      fontSize: '0.95rem',
                      fontWeight: isActive ? 600 : 500,
                    }}
                    secondaryTypographyProps={{
                      fontSize: '0.8rem',
                      color: isActive ? 'rgba(255,255,255,0.7)' : 'text.secondary',
                    }}
                  />
                </ListItemButton>
              </ListItem>
            );
          })}
        </List>
      </Box>

      <Divider />

      {/* Footer */}
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography variant="caption" color="text.secondary">
          Vasanth Experiments v1.0
        </Typography>
      </Box>
    </Box>
  );

  return (
    <Box
      component="nav"
      sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
    >
      {/* Mobile drawer */}
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={onDrawerToggle}
        ModalProps={{
          keepMounted: true, // Better mobile performance
        }}
        sx={{
          display: { xs: 'block', md: 'none' },
          '& .MuiDrawer-paper': {
            boxSizing: 'border-box',
            width: drawerWidth,
          },
        }}
      >
        {drawerContent}
      </Drawer>

      {/* Desktop drawer */}
      <Drawer
        variant="permanent"
        sx={{
          display: { xs: 'none', md: 'block' },
          '& .MuiDrawer-paper': {
            boxSizing: 'border-box',
            width: drawerWidth,
            borderRight: `1px solid ${theme.palette.divider}`,
          },
        }}
        open
      >
        {drawerContent}
      </Drawer>
    </Box>
  );
};

export default Sidebar;