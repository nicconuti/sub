/**
 * Professional Audio Software Layout
 * Main application layout with header, sidebar, and canvas area
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Drawer,
  IconButton,
  Typography,
  useTheme,
  Divider,
  Tooltip,
  Badge,
  Chip,
  alpha,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Settings as SettingsIcon,
  CloudSync as CloudSyncIcon,
  Notifications as NotificationsIcon,
  Help as HelpIcon,
  AccountCircle as AccountIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  TuneRounded as TuneIcon,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';

import { useSimulationStore } from '../../store/simulation';
import { useConnectionStore } from '../../store/connection';
// Temporary constants
const UI_CONSTANTS = {
  SIDEBAR_WIDTH: 320,
  HEADER_HEIGHT: 64,
  TAB_HEIGHT: 48,
};
import { ConnectionIndicator } from '../common/ConnectionIndicator';
import { SimulationProgress } from '../common/SimulationProgress';
import { ControlPanel } from '../panels/ControlPanel';

interface AppLayoutProps {
  children: React.ReactNode;
}

export const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const theme = useTheme();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  
  // Store hooks
  const { 
    currentProject, 
    isSimulating, 
    simulationProgress, 
    startSimulation, 
    stopSimulation 
  } = useSimulationStore();
  
  const { 
    isConnected, 
    connectionError, 
    serverVersion 
  } = useConnectionStore();

  const handleSidebarToggle = useCallback(() => {
    setSidebarOpen(!sidebarOpen);
  }, [sidebarOpen]);

  const handleSimulationToggle = useCallback(() => {
    if (isSimulating) {
      stopSimulation();
    } else {
      startSimulation();
    }
  }, [isSimulating, startSimulation, stopSimulation]);

  const sidebarWidth = sidebarOpen ? UI_CONSTANTS.SIDEBAR_WIDTH : 0;

  return (
    <Box sx={{ display: 'flex', height: '100vh', bgcolor: 'background.default' }}>
      {/* Professional Header */}
      <AppBar 
        position="fixed" 
        sx={{ 
          zIndex: theme.zIndex.drawer + 1,
          bgcolor: 'background.paper',
          borderBottom: `1px solid ${theme.palette.divider}`,
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)',
        }}
      >
        <Toolbar sx={{ minHeight: UI_CONSTANTS.HEADER_HEIGHT }}>
          {/* Sidebar Toggle */}
          <Tooltip title={sidebarOpen ? "Hide Control Panel" : "Show Control Panel"}>
            <IconButton
              edge="start"
              onClick={handleSidebarToggle}
              sx={{ 
                mr: 2,
                color: 'text.primary',
                '&:hover': {
                  bgcolor: alpha(theme.palette.primary.main, 0.1),
                }
              }}
            >
              <MenuIcon />
            </IconButton>
          </Tooltip>

          {/* App Title & Project Info */}
          <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
            <Typography 
              variant="h6" 
              sx={{ 
                fontWeight: 600,
                background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                color: 'transparent',
                mr: 3,
              }}
            >
              Subwoofer Simulation Pro
            </Typography>
            
            {currentProject && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Divider orientation="vertical" flexItem sx={{ mx: 1, height: 24 }} />
                <Typography variant="body2" color="text.secondary">
                  Project:
                </Typography>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  {currentProject.name}
                </Typography>
                <Chip 
                  label={`${currentProject.sources.length} sources`}
                  size="small"
                  variant="outlined"
                  sx={{ 
                    height: 20,
                    fontSize: '0.6875rem',
                    borderColor: 'primary.main',
                    color: 'primary.main',
                  }}
                />
              </Box>
            )}
          </Box>

          {/* Simulation Controls */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mr: 2 }}>
            {isSimulating && (
              <SimulationProgress 
                progress={simulationProgress} 
                variant="compact" 
              />
            )}
            
            <Tooltip title={isSimulating ? "Stop Simulation" : "Start Simulation"}>
              <IconButton
                onClick={handleSimulationToggle}
                disabled={!currentProject || !isConnected}
                sx={{
                  color: isSimulating ? 'error.main' : 'success.main',
                  bgcolor: isSimulating 
                    ? alpha(theme.palette.error.main, 0.1)
                    : alpha(theme.palette.success.main, 0.1),
                  '&:hover': {
                    bgcolor: isSimulating 
                      ? alpha(theme.palette.error.main, 0.2)
                      : alpha(theme.palette.success.main, 0.2),
                  },
                  '&.Mui-disabled': {
                    color: 'action.disabled',
                    bgcolor: 'transparent',
                  }
                }}
              >
                {isSimulating ? <StopIcon /> : <PlayIcon />}
              </IconButton>
            </Tooltip>
          </Box>

          {/* Status & Actions */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {/* Connection Status */}
            <ConnectionIndicator 
              isConnected={isConnected}
              error={connectionError}
              serverVersion={serverVersion}
            />

            {/* Optimization Status */}
            <Tooltip title="Optimization Controls">
              <IconButton
                sx={{ color: 'text.secondary' }}
              >
                <TuneIcon />
              </IconButton>
            </Tooltip>

            {/* Notifications */}
            <Tooltip title="Notifications">
              <IconButton sx={{ color: 'text.secondary' }}>
                <Badge badgeContent={0} color="primary">
                  <NotificationsIcon />
                </Badge>
              </IconButton>
            </Tooltip>

            {/* Settings */}
            <Tooltip title="Settings">
              <IconButton
                onClick={() => setSettingsOpen(true)}
                sx={{ color: 'text.secondary' }}
              >
                <SettingsIcon />
              </IconButton>
            </Tooltip>

            {/* Help */}
            <Tooltip title="Help & Documentation">
              <IconButton sx={{ color: 'text.secondary' }}>
                <HelpIcon />
              </IconButton>
            </Tooltip>

            {/* User Account */}
            <Tooltip title="User Account">
              <IconButton sx={{ color: 'text.secondary' }}>
                <AccountIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Professional Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: UI_CONSTANTS.SIDEBAR_WIDTH, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeInOut' }}
            style={{ overflow: 'hidden' }}
          >
            <Drawer
              variant="permanent"
              sx={{
                width: UI_CONSTANTS.SIDEBAR_WIDTH,
                flexShrink: 0,
                '& .MuiDrawer-paper': {
                  width: UI_CONSTANTS.SIDEBAR_WIDTH,
                  boxSizing: 'border-box',
                  bgcolor: 'background.paper',
                  borderRight: `1px solid ${theme.palette.divider}`,
                  top: UI_CONSTANTS.HEADER_HEIGHT,
                  height: `calc(100vh - ${UI_CONSTANTS.HEADER_HEIGHT}px)`,
                  overflowY: 'auto',
                  overflowX: 'hidden',
                  
                  // Professional scrollbar styling
                  '&::-webkit-scrollbar': {
                    width: 6,
                  },
                  '&::-webkit-scrollbar-track': {
                    bgcolor: 'transparent',
                  },
                  '&::-webkit-scrollbar-thumb': {
                    bgcolor: alpha(theme.palette.text.secondary, 0.3),
                    borderRadius: 3,
                    '&:hover': {
                      bgcolor: alpha(theme.palette.text.secondary, 0.5),
                    },
                  },
                },
              }}
            >
              <ControlPanel />
            </Drawer>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content Area */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          pt: `${UI_CONSTANTS.HEADER_HEIGHT}px`,
          ml: 0, // No margin since drawer is positioned absolute
          width: `calc(100vw - ${sidebarWidth}px)`,
          height: `calc(100vh - ${UI_CONSTANTS.HEADER_HEIGHT}px)`,
          bgcolor: 'background.default',
          position: 'relative',
          overflow: 'hidden',
          transition: theme.transitions.create(['width'], {
            easing: theme.transitions.easing.easeInOut,
            duration: theme.transitions.duration.short,
          }),
        }}
      >
        {/* Content with professional styling */}
        <Box
          sx={{
            width: '100%',
            height: '100%',
            position: 'relative',
            bgcolor: 'background.default',
            
            // Professional grid background for technical feel
            backgroundImage: `
              radial-gradient(circle at 1px 1px, ${alpha(theme.palette.primary.main, 0.15)} 1px, transparent 0)
            `,
            backgroundSize: '20px 20px',
          }}
        >
          {children}
        </Box>
      </Box>

      {/* Settings Dialog */}
      {/* Settings dialog implementation would go here */}
    </Box>
  );
};

export default AppLayout;