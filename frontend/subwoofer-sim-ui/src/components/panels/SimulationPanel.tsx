/**
 * Simulation Parameters Panel
 * Professional controls for simulation settings
 */

import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  TextField,
  Slider,
  FormControlLabel,
  Switch,
  Button,
  Divider,
  Chip,
  useTheme,
  alpha,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';

import { useSimulationStore } from '../../store/simulation';

// Temporary constants
const SIMULATION_LIMITS = {
  FREQUENCY: { MIN: 20, MAX: 20000 },
  SPL_RMS: { MIN: 50, MAX: 150 },
  GAIN_DB: { MIN: -60, MAX: 20 },
  DELAY_MS: { MIN: 0, MAX: 1000 },
  ANGLE: { MIN: 0, MAX: 360 },
  GRID_RESOLUTION: { MIN: 0.01, MAX: 1.0 },
  SPEED_OF_SOUND: { MIN: 300, MAX: 400 },
  MAX_SOURCES: 100,
  MAX_ROOM_VERTICES: 50,
};

export const SimulationPanel: React.FC = () => {
  const theme = useTheme();
  const { 
    simulationParams, 
    updateSimulationParams, 
    isSimulating,
    startSimulation,
    stopSimulation
  } = useSimulationStore();

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {/* Simulation Control */}
      <Card sx={{ bgcolor: 'background.paper' }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <SettingsIcon sx={{ color: 'primary.main' }} />
            <Typography variant="h6">Simulation Control</Typography>
          </Box>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              variant={isSimulating ? "outlined" : "contained"}
              color={isSimulating ? "error" : "primary"}
              startIcon={isSimulating ? <StopIcon /> : <PlayIcon />}
              onClick={isSimulating ? stopSimulation : startSimulation}
              fullWidth
            >
              {isSimulating ? 'Stop Simulation' : 'Start Simulation'}
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Frequency Settings */}
      <Card>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Frequency Settings
          </Typography>
          
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Frequency (Hz)
            </Typography>
            <Slider
              value={simulationParams.frequency}
              onChange={(_, value) => updateSimulationParams({ frequency: value as number })}
              min={SIMULATION_LIMITS.FREQUENCY.MIN}
              max={SIMULATION_LIMITS.FREQUENCY.MAX}
              step={1}
              valueLabelDisplay="auto"
              marks={[
                { value: 20, label: '20' },
                { value: 80, label: '80' },
                { value: 200, label: '200' },
                { value: 1000, label: '1k' },
                { value: 20000, label: '20k' },
              ]}
              sx={{
                '& .MuiSlider-markLabel': {
                  fontSize: '0.75rem',
                },
              }}
            />
          </Box>

          <TextField
            label="Frequency"
            type="number"
            value={simulationParams.frequency}
            onChange={(e) => updateSimulationParams({ frequency: parseFloat(e.target.value) })}
            InputProps={{
              endAdornment: <Typography variant="caption">Hz</Typography>,
            }}
            size="small"
            fullWidth
          />
        </CardContent>
      </Card>

      {/* Acoustic Parameters */}
      <Card>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Acoustic Parameters
          </Typography>
          
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <TextField
              label="Speed of Sound"
              type="number"
              value={simulationParams.speed_of_sound}
              onChange={(e) => updateSimulationParams({ speed_of_sound: parseFloat(e.target.value) })}
              InputProps={{
                endAdornment: <Typography variant="caption">m/s</Typography>,
              }}
              size="small"
              helperText="Typical: 343 m/s (20°C)"
            />

            <TextField
              label="Grid Resolution"
              type="number"
              value={simulationParams.grid_resolution}
              onChange={(e) => updateSimulationParams({ grid_resolution: parseFloat(e.target.value) })}
              InputProps={{
                endAdornment: <Typography variant="caption">m</Typography>,
              }}
              size="small"
              helperText="Lower values = higher accuracy"
            />
          </Box>
        </CardContent>
      </Card>

      {/* Room Configuration */}
      <Card>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Room Configuration
          </Typography>
          
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Room Vertices: {simulationParams.room_vertices.length}
            </Typography>
            
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              {simulationParams.room_vertices.map((vertex, index) => (
                <Chip
                  key={index}
                  label={`(${vertex[0]}, ${vertex[1]})`}
                  size="small"
                  variant="outlined"
                  sx={{ fontFamily: 'mono', fontSize: '0.75rem' }}
                />
              ))}
            </Box>
          </Box>

          <Button variant="outlined" size="small" fullWidth>
            Edit Room Geometry
          </Button>
        </CardContent>
      </Card>
    </Box>
  );
};