/**
 * Professional SPL Visualization Component
 * High-performance acoustic data visualization with Plotly.js
 */

import React, { useMemo, useCallback, useEffect, useState } from 'react';
import { Box, Chip, Tooltip, IconButton, Menu, MenuItem, useTheme, Typography, Card } from '@mui/material';
import {
  ColorLens as ColorLensIcon,
  Settings as SettingsIcon,
  Fullscreen as FullscreenIcon,
  GetApp as ExportIcon,
} from '@mui/icons-material';

// Import react-plotly.js properly
import Plot from 'react-plotly.js';
import type { Layout, Config, PlotMarker } from 'plotly.js';

// Temporary type definitions
interface SplMapData {
  X: number[][];
  Y: number[][];
  SPL: number[][];
  frequency: number;
  timestamp: number;
  statistics?: any;
}

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

interface ViewState {
  zoom: number;
  pan: [number, number];
  center: [number, number];
  rotation: number;
}
import { audioColors, getSplColor } from '../../theme/audioTheme';
import { useSimulationStore } from '../../store/simulation';

interface SplViewerProps {
  splData?: SplMapData;
  sources: SourceData[];
  viewState: ViewState;
  onViewStateChange: (viewState: ViewState) => void;
  onSourceSelect?: (sourceId: string) => void;
  onPointHover?: (point: [number, number]) => void;
  showSources?: boolean;
  showGrid?: boolean;
  interactive?: boolean;
  colorscale?: string;
  splRange?: [number, number];
}

// Professional audio colorscales
const AUDIO_COLORSCALES = {
  'spectrum': [
    [0, '#000428'],      // Deep blue (quiet)
    [0.2, '#004e92'],    // Blue
    [0.4, '#009ffd'],    // Light blue  
    [0.6, '#00d2ff'],    // Cyan
    [0.8, '#ffa400'],    // Orange
    [1, '#ff0000']       // Red (loud)
  ],
  'thermal': [
    [0, '#000000'],      // Black
    [0.25, '#440154'],   // Dark purple
    [0.5, '#21908c'],    // Teal
    [0.75, '#fde725'],   // Yellow
    [1, '#ffffff']       // White
  ],
  'acoustic': [
    [0, '#0d1421'],      // Near black
    [0.1, '#1a365d'],    // Dark blue
    [0.3, '#2d5a87'],    // Medium blue
    [0.5, '#4f8db8'],    // Light blue
    [0.7, '#00d4ff'],    // Cyan (primary)
    [0.9, '#ff6b35'],    // Orange (secondary)
    [1, '#ff0000']       // Red (critical)
  ],
  'professional': [
    [0, '#000000'],      // Black (silence)
    [0.15, '#1a1a2e'],   // Dark gray
    [0.3, '#16213e'],    // Dark blue
    [0.45, '#0f3460'],   // Medium blue
    [0.6, '#00d4ff'],    // Cyan (optimal)
    [0.75, '#ff9500'],   // Orange (warning)
    [0.9, '#ff4757'],    // Red (danger)
    [1, '#ffffff']       // White (extreme)
  ]
};

export const SplViewer: React.FC<SplViewerProps> = ({
  splData,
  sources = [],
  viewState,
  onViewStateChange,
  onSourceSelect,
  onPointHover,
  showSources = true,
  showGrid = true,
  interactive = true,
  colorscale = 'acoustic',
  splRange,
}) => {
  const theme = useTheme();
  const { currentProject } = useSimulationStore();
  
  const [colorMenuAnchor, setColorMenuAnchor] = useState<null | HTMLElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Calculate SPL range automatically if not provided
  const calculatedSplRange = useMemo(() => {
    if (splRange) return splRange;
    if (!splData) return [60, 120];
    
    const flatSpl = splData.SPL.flat();
    const minSpl = Math.min(...flatSpl);
    const maxSpl = Math.max(...flatSpl);
    const padding = (maxSpl - minSpl) * 0.1;
    
    return [
      Math.max(0, minSpl - padding),
      Math.min(150, maxSpl + padding)
    ] as [number, number];
  }, [splData, splRange]);

  // Professional plot data preparation
  const plotData: any[] = useMemo(() => {
    const data: any[] = [];

    // SPL Heatmap
    if (splData) {
      data.push({
        type: 'heatmap',
        x: splData.X[0], // First row for x-axis
        y: splData.Y.map(row => row[0]), // First column for y-axis
        z: splData.SPL,
        colorscale: AUDIO_COLORSCALES[colorscale as keyof typeof AUDIO_COLORSCALES] || AUDIO_COLORSCALES.acoustic,
        zmin: calculatedSplRange[0],
        zmax: calculatedSplRange[1],
        hoverongaps: false,
        hovertemplate: 
          '<b>SPL:</b> %{z:.1f} dB<br>' +
          '<b>Position:</b> (%{x:.2f}, %{y:.2f}) m<br>' +
          '<b>Frequency:</b> ' + (splData.frequency || 80) + ' Hz<br>' +
          '<extra></extra>',
        colorbar: {
          title: {
            text: 'SPL (dB)',
            font: {
              family: theme.typography.fontFamily,
              size: 12,
              color: theme.palette.text.primary,
            }
          },
          tickfont: {
            family: theme.typography.fontFamily,
            size: 10,
            color: theme.palette.text.secondary,
          },
          thickness: 15,
          len: 0.8,
          x: 1.02,
          bgcolor: 'rgba(0,0,0,0)',
          bordercolor: theme.palette.divider,
          borderwidth: 1,
          tickmode: 'linear',
          tick0: calculatedSplRange[0],
          dtick: (calculatedSplRange[1] - calculatedSplRange[0]) / 10,
        },
        showscale: true,
      });
    }

    // Source Markers
    if (showSources && sources.length > 0) {
      const enabledSources = sources.filter(s => s.enabled !== false);
      
      if (enabledSources.length > 0) {
        data.push({
          type: 'scatter',
          mode: 'markers+text',
          x: enabledSources.map(s => s.x),
          y: enabledSources.map(s => s.y),
          text: enabledSources.map(s => s.name || s.id),
          textposition: 'top center',
          textfont: {
            family: theme.typography.fontFamily,
            size: 10,
            color: theme.palette.text.primary,
          },
          marker: {
            size: enabledSources.map(s => Math.max(8, Math.min(20, s.spl_rms / 8))),
            color: enabledSources.map(s => s.color || audioColors.primary.main),
            symbol: 'circle',
            line: {
              color: theme.palette.background.paper,
              width: 2,
            },
            opacity: 0.9,
          } as Partial<PlotMarker>,
          hovertemplate: 
            '<b>%{text}</b><br>' +
            '<b>Position:</b> (%{x:.2f}, %{y:.2f}) m<br>' +
            '<b>SPL RMS:</b> %{customdata[0]:.1f} dB<br>' +
            '<b>Gain:</b> %{customdata[1]:+.1f} dB<br>' +
            '<b>Delay:</b> %{customdata[2]:.1f} ms<br>' +
            '<b>Angle:</b> %{customdata[3]:.0f}°<br>' +
            '<extra></extra>',
          customdata: enabledSources.map(s => [s.spl_rms, s.gain_db, s.delay_ms, s.angle]),
          name: 'Sources',
        });
      }
    }

    // Room Boundaries
    if (currentProject?.simulation_params.room_vertices) {
      const vertices = currentProject.simulation_params.room_vertices;
      const closedVertices = [...vertices, vertices[0]]; // Close the polygon
      
      data.push({
        type: 'scatter',
        mode: 'lines',
        x: closedVertices.map(v => v[0]),
        y: closedVertices.map(v => v[1]),
        line: {
          color: audioColors.text.secondary,
          width: 2,
          dash: 'dash',
        },
        name: 'Room Boundary',
        hoverinfo: 'skip',
        showlegend: false,
      });
    }

    return data;
  }, [splData, sources, showSources, currentProject, colorscale, calculatedSplRange, theme]);

  // Professional layout configuration
  const layout: Partial<Layout> = useMemo(() => ({
    title: {
      text: splData ? `SPL Map - ${splData.frequency} Hz` : 'SPL Visualization',
      font: {
        family: theme.typography.fontFamily,
        size: 16,
        color: theme.palette.text.primary,
      },
      x: 0.05,
      y: 0.95,
    },
    xaxis: {
      title: {
        text: 'X Position (m)',
        font: {
          family: theme.typography.fontFamily,
          size: 12,
          color: theme.palette.text.primary,
        }
      },
      tickfont: {
        family: theme.typography.fontFamily,
        size: 10,
        color: theme.palette.text.secondary,
      },
      gridcolor: showGrid ? audioColors.border : 'transparent',
      gridwidth: 1,
      zeroline: false,
      autorange: false,
      range: [viewState.center[0] - 5/viewState.zoom, viewState.center[0] + 5/viewState.zoom],
    },
    yaxis: {
      title: {
        text: 'Y Position (m)',
        font: {
          family: theme.typography.fontFamily,
          size: 12,
          color: theme.palette.text.primary,
        }
      },
      tickfont: {
        family: theme.typography.fontFamily,
        size: 10,
        color: theme.palette.text.secondary,
      },
      gridcolor: showGrid ? audioColors.border : 'transparent',
      gridwidth: 1,
      zeroline: false,
      autorange: false,
      range: [viewState.center[1] - 5/viewState.zoom, viewState.center[1] + 5/viewState.zoom],
      scaleanchor: 'x', // Maintain aspect ratio
      scaleratio: 1,
    },
    plot_bgcolor: 'transparent',
    paper_bgcolor: 'transparent',
    font: {
      family: theme.typography.fontFamily,
      color: theme.palette.text.primary,
    },
    margin: { l: 60, r: 80, t: 60, b: 60 },
    autosize: true,
    showlegend: false,
    hovermode: 'closest',
  }), [splData, theme, showGrid, viewState, audioColors]);

  // Professional plot configuration
  const config: Partial<Config> = {
    displayModeBar: interactive,
    displaylogo: false,
    modeBarButtonsToRemove: [
      'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d',
      'autoScale2d', 'resetScale2d', 'hoverClosestCartesian',
      'hoverCompareCartesian', 'toggleSpikelines'
    ],
    modeBarButtonsToAdd: [],
    responsive: true,
    scrollZoom: interactive,
    doubleClick: false,
    showTips: false,
  };

  // Handle plot interactions
  const handlePlotlyClick = useCallback((data: any) => {
    if (!interactive) return;
    
    const point = data.points?.[0];
    if (point) {
      // Check if clicking on a source
      if (point.data.name === 'Sources') {
        const sourceIndex = point.pointIndex;
        const enabledSources = sources.filter(s => s.enabled !== false);
        if (sourceIndex < enabledSources.length && onSourceSelect) {
          onSourceSelect(enabledSources[sourceIndex].id);
        }
      }
    }
  }, [interactive, sources, onSourceSelect]);

  const handlePlotlyHover = useCallback((data: any) => {
    if (!interactive || !onPointHover) return;
    
    const point = data.points?.[0];
    if (point && point.data.type === 'heatmap') {
      onPointHover([point.x, point.y]);
    }
  }, [interactive, onPointHover]);

  const handlePlotlyRelayout = useCallback((eventData: any) => {
    if (!interactive || !onViewStateChange) return;
    
    // Update view state based on zoom/pan
    if (eventData['xaxis.range[0]'] !== undefined) {
      const xRange = [eventData['xaxis.range[0]'], eventData['xaxis.range[1]']];
      const yRange = [eventData['yaxis.range[0]'], eventData['yaxis.range[1]']];
      
      const centerX = (xRange[0] + xRange[1]) / 2;
      const centerY = (yRange[0] + yRange[1]) / 2;
      const zoom = 10 / (xRange[1] - xRange[0]); // Approximate zoom calculation
      
      onViewStateChange({
        ...viewState,
        center: [centerX, centerY],
        zoom: Math.max(0.1, Math.min(10, zoom)),
      });
    }
  }, [interactive, onViewStateChange, viewState]);

  return (
    <Box sx={{ 
      width: '100%', 
      height: '100%', 
      position: 'relative',
      bgcolor: 'background.default',
    }}>
      {/* Control Toolbar */}
      {interactive && (
        <Box sx={{
          position: 'absolute',
          top: 16,
          right: 16,
          zIndex: 10,
          display: 'flex',
          gap: 1,
          bgcolor: 'background.paper',
          borderRadius: 1,
          border: `1px solid ${theme.palette.divider}`,
          p: 0.5,
        }}>
          {/* SPL Range Indicator */}
          {splData && (
            <Chip
              label={`${calculatedSplRange[0].toFixed(0)}-${calculatedSplRange[1].toFixed(0)} dB`}
              size="small"
              sx={{ 
                bgcolor: 'background.surface',
                color: 'text.primary',
                fontFamily: 'mono',
                fontSize: '0.75rem',
              }}
            />
          )}

          {/* Colorscale Selector */}
          <Tooltip title="Change Colorscale">
            <IconButton
              size="small"
              onClick={(e) => setColorMenuAnchor(e.currentTarget)}
            >
              <ColorLensIcon />
            </IconButton>
          </Tooltip>

          {/* Settings */}
          <Tooltip title="Plot Settings">
            <IconButton size="small">
              <SettingsIcon />
            </IconButton>
          </Tooltip>

          {/* Export */}
          <Tooltip title="Export Plot">
            <IconButton size="small">
              <ExportIcon />
            </IconButton>
          </Tooltip>

          {/* Fullscreen */}
          <Tooltip title="Fullscreen">
            <IconButton 
              size="small"
              onClick={() => setIsFullscreen(!isFullscreen)}
            >
              <FullscreenIcon />
            </IconButton>
          </Tooltip>
        </Box>
      )}

      {/* Main Plot */}
      <Plot
        data={plotData}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
        onClick={handlePlotlyClick}
        onHover={handlePlotlyHover}
        onRelayout={handlePlotlyRelayout}
      />

      {/* Colorscale Menu */}
      <Menu
        anchorEl={colorMenuAnchor}
        open={Boolean(colorMenuAnchor)}
        onClose={() => setColorMenuAnchor(null)}
        PaperProps={{
          sx: {
            bgcolor: 'background.paper',
            border: `1px solid ${theme.palette.divider}`,
          }
        }}
      >
        {Object.keys(AUDIO_COLORSCALES).map((scale) => (
          <MenuItem
            key={scale}
            onClick={() => {
              // Handle colorscale change
              setColorMenuAnchor(null);
            }}
            selected={scale === colorscale}
          >
            {scale.charAt(0).toUpperCase() + scale.slice(1)}
          </MenuItem>
        ))}
      </Menu>
    </Box>
  );
};

export default SplViewer;