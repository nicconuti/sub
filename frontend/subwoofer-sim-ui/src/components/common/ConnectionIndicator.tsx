/**
 * Professional Connection Status Indicator
 * Shows WebSocket connection status with visual feedback
 */

import React from 'react';
import {
  Box,
  Chip,
  Tooltip,
  Typography,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Wifi as ConnectedIcon,
  WifiOff as DisconnectedIcon,
  Sync as ConnectingIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

interface ConnectionIndicatorProps {
  isConnected: boolean;
  error?: string | null;
  serverVersion?: string;
  reconnectAttempts?: number;
  compact?: boolean;
}

export const ConnectionIndicator: React.FC<ConnectionIndicatorProps> = ({
  isConnected,
  error,
  serverVersion,
  reconnectAttempts = 0,
  compact = false,
}) => {
  const theme = useTheme();

  const getConnectionStatus = () => {
    if (error) return 'error';
    if (!isConnected && reconnectAttempts > 0) return 'reconnecting';
    if (!isConnected) return 'disconnected';
    return 'connected';
  };

  const status = getConnectionStatus();

  const statusConfig = {
    connected: {
      icon: ConnectedIcon,
      color: theme.palette.success.main,
      bgColor: alpha(theme.palette.success.main, 0.1),
      label: 'Connected',
      message: `Connected to server${serverVersion ? ` v${serverVersion}` : ''}`,
    },
    disconnected: {
      icon: DisconnectedIcon,
      color: theme.palette.text.disabled,
      bgColor: alpha(theme.palette.text.disabled, 0.1),
      label: 'Offline',
      message: 'Not connected to server',
    },
    reconnecting: {
      icon: ConnectingIcon,
      color: theme.palette.warning.main,
      bgColor: alpha(theme.palette.warning.main, 0.1),
      label: 'Reconnecting',
      message: `Reconnecting... (attempt ${reconnectAttempts})`,
    },
    error: {
      icon: ErrorIcon,
      color: theme.palette.error.main,
      bgColor: alpha(theme.palette.error.main, 0.1),
      label: 'Error',
      message: error || 'Connection error',
    },
  };

  const config = statusConfig[status];
  const IconComponent = config.icon;

  if (compact) {
    return (
      <Tooltip title={config.message}>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            p: 0.5,
            borderRadius: 1,
            bgcolor: config.bgColor,
            border: `1px solid ${alpha(config.color, 0.3)}`,
          }}
        >
          <motion.div
            animate={status === 'reconnecting' ? { rotate: 360 } : {}}
            transition={{ repeat: status === 'reconnecting' ? Infinity : 0, duration: 1 }}
          >
            <IconComponent 
              sx={{ 
                fontSize: 16, 
                color: config.color,
              }} 
            />
          </motion.div>
        </Box>
      </Tooltip>
    );
  }

  return (
    <Tooltip 
      title={
        <Box>
          <Typography variant="body2" sx={{ fontWeight: 500 }}>
            {config.message}
          </Typography>
          {serverVersion && (
            <Typography variant="caption" color="text.secondary">
              Server Version: {serverVersion}
            </Typography>
          )}
          {reconnectAttempts > 0 && (
            <Typography variant="caption" color="text.secondary">
              Reconnect Attempts: {reconnectAttempts}
            </Typography>
          )}
        </Box>
      }
    >
      <Chip
        icon={
          <motion.div
            animate={status === 'reconnecting' ? { rotate: 360 } : {}}
            transition={{ repeat: status === 'reconnecting' ? Infinity : 0, duration: 1 }}
          >
            <IconComponent />
          </motion.div>
        }
        label={config.label}
        size="small"
        sx={{
          color: config.color,
          bgcolor: config.bgColor,
          border: `1px solid ${alpha(config.color, 0.3)}`,
          '& .MuiChip-icon': {
            color: config.color,
          },
          fontWeight: 500,
          fontSize: '0.75rem',
          height: 24,
        }}
      />
    </Tooltip>
  );
};

export default ConnectionIndicator;