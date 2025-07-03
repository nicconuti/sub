/**
 * Sources Management Panel
 * Professional controls for subwoofer source configuration
 */

import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Chip,
  TextField,
  Slider,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  GraphicEq as SourceIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
} from '@mui/icons-material';

import { useSimulationStore, useCurrentSources, useSelectedSources } from '../../store/simulation';
import { getSplColor } from '../../theme/audioTheme';

// Temporary type definition
interface SourceData {
  id: string;
  x: number;
  y: number;
  spl_rms: number;
  gain_db: number;
  delay_ms: number;
  angle: number;
  polarity: -1 | 1;
  name?: string;
  color?: string;
  enabled?: boolean;
}

export const SourcesPanel: React.FC = () => {
  const theme = useTheme();
  const sources = useCurrentSources();
  const selectedSources = useSelectedSources();
  const { addSource, updateSource, removeSource, selectSources } = useSimulationStore();

  const createNewSource = () => {
    const newSource: SourceData = {
      id: `source_${Date.now()}`,
      x: 5,
      y: 2,
      spl_rms: 105,
      gain_db: 0,
      delay_ms: 0,
      angle: 0,
      polarity: 1,
      name: `Source ${sources.length + 1}`,
      enabled: true,
    };
    addSource(newSource);
  };

  const toggleSourceVisibility = (sourceId: string) => {
    const source = sources.find(s => s.id === sourceId);
    if (source) {
      updateSource(sourceId, { enabled: !source.enabled });
    }
  };

  const handleSourceSelect = (sourceId: string) => {
    const isSelected = selectedSources.some(s => s.id === sourceId);
    if (isSelected) {
      selectSources(selectedSources.filter(s => s.id !== sourceId).map(s => s.id));
    } else {
      selectSources([...selectedSources.map(s => s.id), sourceId]);
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {/* Sources Header */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <SourceIcon sx={{ color: 'primary.main' }} />
              <Typography variant="h6">Sources</Typography>
              <Chip
                label={sources.length}
                size="small"
                color="primary"
                sx={{ fontWeight: 600 }}
              />
            </Box>
            
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={createNewSource}
              size="small"
            >
              Add Source
            </Button>
          </Box>

          {selectedSources.length > 0 && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                {selectedSources.length} source(s) selected
              </Typography>
              <Button
                variant="outlined"
                size="small"
                onClick={() => selectSources([])}
              >
                Clear Selection
              </Button>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Sources List */}
      <Card>
        <List dense>
          {sources.map((source, index) => (
            <ListItem
              key={source.id}
              sx={{
                border: selectedSources.some(s => s.id === source.id) 
                  ? `2px solid ${theme.palette.primary.main}`
                  : `1px solid ${theme.palette.divider}`,
                borderRadius: 1,
                mb: 1,
                bgcolor: source.enabled 
                  ? 'background.paper' 
                  : alpha(theme.palette.action.disabled, 0.1),
                cursor: 'pointer',
                '&:hover': {
                  bgcolor: alpha(theme.palette.primary.main, 0.04),
                },
              }}
              onClick={() => handleSourceSelect(source.id)}
            >
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {source.name || source.id}
                    </Typography>
                    <Chip
                      label={`${source.spl_rms.toFixed(0)} dB`}
                      size="small"
                      sx={{
                        bgcolor: alpha(getSplColor(source.spl_rms), 0.2),
                        color: getSplColor(source.spl_rms),
                        fontFamily: 'mono',
                        fontSize: '0.6875rem',
                        height: 18,
                      }}
                    />
                  </Box>
                }
                secondary={
                  <Box sx={{ mt: 0.5 }}>
                    <Typography variant="caption" color="text.secondary">
                      Position: ({source.x.toFixed(2)}, {source.y.toFixed(2)}) m
                    </Typography>
                    <br />
                    <Typography variant="caption" color="text.secondary">
                      Gain: {source.gain_db >= 0 ? '+' : ''}{source.gain_db.toFixed(1)} dB | Delay: {source.delay_ms.toFixed(1)} ms
                    </Typography>
                  </Box>
                }
              />
              
              <ListItemSecondaryAction>
                <Box sx={{ display: 'flex', gap: 0.5 }}>
                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleSourceVisibility(source.id);
                    }}
                    sx={{ 
                      color: source.enabled ? 'primary.main' : 'action.disabled',
                    }}
                  >
                    {source.enabled ? <VisibilityIcon /> : <VisibilityOffIcon />}
                  </IconButton>
                  
                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      // Handle edit
                    }}
                  >
                    <EditIcon />
                  </IconButton>
                  
                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      removeSource(source.id);
                    }}
                    sx={{ color: 'error.main' }}
                  >
                    <DeleteIcon />
                  </IconButton>
                </Box>
              </ListItemSecondaryAction>
            </ListItem>
          ))}
        </List>
        
        {sources.length === 0 && (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              No sources added yet
            </Typography>
            <Button
              variant="outlined"
              startIcon={<AddIcon />}
              onClick={createNewSource}
              sx={{ mt: 1 }}
            >
              Add First Source
            </Button>
          </Box>
        )}
      </Card>

      {/* Source Editor */}
      {selectedSources.length === 1 && (
        <Card>
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>
              Edit Source: {selectedSources[0].name}
            </Typography>
            
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <TextField
                label="Name"
                value={selectedSources[0].name || ''}
                onChange={(e) => updateSource(selectedSources[0].id, { name: e.target.value })}
                size="small"
                fullWidth
              />
              
              <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                  label="X Position"
                  type="number"
                  value={selectedSources[0].x}
                  onChange={(e) => updateSource(selectedSources[0].id, { x: parseFloat(e.target.value) })}
                  InputProps={{
                    endAdornment: <Typography variant="caption">m</Typography>,
                  }}
                  size="small"
                  fullWidth
                />
                <TextField
                  label="Y Position"
                  type="number"
                  value={selectedSources[0].y}
                  onChange={(e) => updateSource(selectedSources[0].id, { y: parseFloat(e.target.value) })}
                  InputProps={{
                    endAdornment: <Typography variant="caption">m</Typography>,
                  }}
                  size="small"
                  fullWidth
                />
              </Box>

              <Box>
                <Typography variant="body2" gutterBottom>
                  SPL RMS: {selectedSources[0].spl_rms.toFixed(1)} dB
                </Typography>
                <Slider
                  value={selectedSources[0].spl_rms}
                  onChange={(_, value) => updateSource(selectedSources[0].id, { spl_rms: value as number })}
                  min={50}
                  max={150}
                  step={0.1}
                  valueLabelDisplay="auto"
                  marks={[
                    { value: 85, label: '85' },
                    { value: 105, label: '105' },
                    { value: 120, label: '120' },
                  ]}
                />
              </Box>

              <Box>
                <Typography variant="body2" gutterBottom>
                  Gain: {selectedSources[0].gain_db >= 0 ? '+' : ''}{selectedSources[0].gain_db.toFixed(1)} dB
                </Typography>
                <Slider
                  value={selectedSources[0].gain_db}
                  onChange={(_, value) => updateSource(selectedSources[0].id, { gain_db: value as number })}
                  min={-20}
                  max={20}
                  step={0.1}
                  valueLabelDisplay="auto"
                  marks={[
                    { value: -10, label: '-10' },
                    { value: 0, label: '0' },
                    { value: 10, label: '+10' },
                  ]}
                />
              </Box>
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};