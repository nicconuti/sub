/**
 * Professional Simulation Progress Indicator
 * Shows real-time simulation progress with professional styling
 */

import React from 'react';
import {
  Box,
  LinearProgress,
  Typography,
  Chip,
  Tooltip,
  useTheme,
  alpha,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  TrendingUp as ProgressIcon,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';

interface SimulationProgressProps {
  progress: {
    progress: number;           // 0-100
    current_step: string;
    estimated_time?: number;    // seconds remaining
    chunks_received?: number;
    total_chunks?: number;
  };
  variant?: 'full' | 'compact' | 'minimal';
  showEstimatedTime?: boolean;
  showChunkProgress?: boolean;
}

export const SimulationProgress: React.FC<SimulationProgressProps> = ({
  progress,
  variant = 'full',
  showEstimatedTime = true,
  showChunkProgress = true,
}) => {
  const theme = useTheme();

  const formatTime = (seconds: number): string => {
    if (seconds < 60) {
      return `${Math.round(seconds)}s`;
    } else if (seconds < 3600) {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = Math.round(seconds % 60);
      return `${minutes}m ${remainingSeconds}s`;
    } else {
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      return `${hours}h ${minutes}m`;
    }
  };

  const getStepDisplayName = (step: string): string => {
    const stepNames: { [key: string]: string } = {
      'idle': 'Ready',
      'initializing': 'Initializing',
      'preparing_sources': 'Preparing Sources',
      'setting_up_grid': 'Setting Up Grid',
      'calculating_spl': 'Calculating SPL',
      'applying_room_mask': 'Applying Room Mask',
      'complete': 'Complete',
      'stopped': 'Stopped',
      'error': 'Error',
    };
    return stepNames[step] || step;
  };

  const getProgressColor = () => {
    if (progress.current_step === 'error') return theme.palette.error.main;
    if (progress.current_step === 'complete') return theme.palette.success.main;
    if (progress.current_step === 'stopped') return theme.palette.text.disabled;
    return theme.palette.primary.main;
  };

  const isActive = !['idle', 'complete', 'stopped', 'error'].includes(progress.current_step);

  if (variant === 'minimal') {
    return (
      <Tooltip title={`${getStepDisplayName(progress.current_step)} - ${progress.progress.toFixed(1)}%`}>
        <Box sx={{ width: 40, height: 4, bgcolor: 'background.surface', borderRadius: 2, overflow: 'hidden' }}>
          <Box
            sx={{
              width: `${progress.progress}%`,
              height: '100%',
              bgcolor: getProgressColor(),
              transition: 'width 0.3s ease-in-out',
            }}
          />
        </Box>
      </Tooltip>
    );
  }

  if (variant === 'compact') {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 120 }}>
        <motion.div
          animate={isActive ? { rotate: 360 } : {}}
          transition={{ repeat: isActive ? Infinity : 0, duration: 2, ease: 'linear' }}
        >
          <ProgressIcon sx={{ fontSize: 16, color: getProgressColor() }} />
        </motion.div>
        
        <Box sx={{ flexGrow: 1, minWidth: 60 }}>
          <LinearProgress
            variant="determinate"
            value={progress.progress}
            sx={{
              height: 4,
              borderRadius: 2,
              bgcolor: alpha(getProgressColor(), 0.2),
              '& .MuiLinearProgress-bar': {
                bgcolor: getProgressColor(),
                borderRadius: 2,
              },
            }}
          />
        </Box>
        
        <Typography variant="caption" sx={{ 
          fontFamily: 'mono', 
          minWidth: 35,
          color: 'text.secondary',
          fontSize: '0.6875rem',
        }}>
          {progress.progress.toFixed(0)}%
        </Typography>
      </Box>
    );
  }

  // Full variant
  return (
    <Box sx={{ 
      width: '100%',
      p: 2,
      bgcolor: 'background.paper',
      borderRadius: 1,
      border: `1px solid ${theme.palette.divider}`,
    }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <motion.div
            animate={isActive ? { rotate: 360 } : {}}
            transition={{ repeat: isActive ? Infinity : 0, duration: 2, ease: 'linear' }}
          >
            <ProgressIcon sx={{ fontSize: 20, color: getProgressColor() }} />
          </motion.div>
          
          <Typography variant="body2" sx={{ fontWeight: 500 }}>
            {getStepDisplayName(progress.current_step)}
          </Typography>
        </Box>

        <Chip
          label={`${progress.progress.toFixed(1)}%`}
          size="small"
          sx={{
            bgcolor: alpha(getProgressColor(), 0.1),
            color: getProgressColor(),
            fontFamily: 'mono',
            fontWeight: 600,
            fontSize: '0.75rem',
            height: 20,
          }}
        />
      </Box>

      {/* Progress Bar */}
      <LinearProgress
        variant="determinate"
        value={progress.progress}
        sx={{
          height: 6,
          borderRadius: 3,
          bgcolor: alpha(getProgressColor(), 0.2),
          mb: 1,
          '& .MuiLinearProgress-bar': {
            bgcolor: getProgressColor(),
            borderRadius: 3,
            transition: 'transform 0.3s ease-in-out',
          },
        }}
      />

      {/* Details */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        {/* Chunk Progress */}
        <AnimatePresence>
          {showChunkProgress && progress.chunks_received !== undefined && progress.total_chunks !== undefined && (
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
            >
              <Typography variant="caption" color="text.secondary">
                Chunks: {progress.chunks_received}/{progress.total_chunks}
              </Typography>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Estimated Time */}
        <AnimatePresence>
          {showEstimatedTime && progress.estimated_time !== undefined && progress.estimated_time > 0 && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
            >
              <Typography variant="caption" color="text.secondary" sx={{ fontFamily: 'mono' }}>
                ETA: {formatTime(progress.estimated_time)}
              </Typography>
            </motion.div>
          )}
        </AnimatePresence>
      </Box>
    </Box>
  );
};

export default SimulationProgress;