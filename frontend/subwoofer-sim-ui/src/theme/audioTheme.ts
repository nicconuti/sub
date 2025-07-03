/**
 * Professional Audio Software Theme
 * Dark theme optimized for acoustic engineering and long working sessions
 */

import { createTheme } from '@mui/material/styles';

// Professional Audio Color Palette
const audioPalette = {
  // Main Brand Colors
  primary: {
    main: '#00D4FF',        // Cyan - typical for audio spectrum displays
    light: '#4AE6FF',       // Light cyan
    dark: '#0099CC',        // Darker cyan
    contrastText: '#000000',
  },
  secondary: {
    main: '#FF6B35',        // Orange - for warnings and important controls
    light: '#FF9066',       // Light orange
    dark: '#CC4A1A',        // Dark orange
    contrastText: '#FFFFFF',
  },
  
  // Advanced Audio-specific Colors
  audio: {
    // SPL Level Colors (dB visualization)
    spl: {
      low: '#2E7D32',       // Green for low SPL
      moderate: '#F57C00',   // Orange for moderate SPL  
      high: '#D32F2F',      // Red for high SPL
      critical: '#7B1FA2',  // Purple for critical levels
    },
    // Frequency Band Colors
    frequency: {
      sub: '#9C27B0',       // Purple for sub frequencies
      bass: '#3F51B5',      // Blue for bass
      mid: '#4CAF50',       // Green for mids
      high: '#FF9800',      // Orange for highs
      ultra: '#F44336',     // Red for ultra-high
    },
    // Phase Colors
    phase: {
      inPhase: '#4CAF50',   // Green for in-phase
      outPhase: '#F44336',  // Red for out-of-phase
      neutral: '#FFC107',   // Amber for neutral
    }
  },
  
  // Professional Dark Background Scheme
  background: {
    default: '#0A0A0A',     // Almost black - professional studio look
    paper: '#1A1A1A',       // Dark gray for panels
    surface: '#2A2A2A',     // Medium gray for cards
    elevated: '#3A3A3A',    // Lighter gray for raised elements
  },
  
  // Text Colors optimized for readability
  text: {
    primary: '#FFFFFF',     // Pure white for main text
    secondary: '#B0B0B0',   // Light gray for secondary text
    disabled: '#666666',    // Medium gray for disabled text
    hint: '#808080',        // Subtle gray for hints
  },
  
  // Action Colors
  action: {
    active: '#00D4FF',      // Cyan for active states
    hover: 'rgba(0, 212, 255, 0.08)',  // Subtle cyan hover
    selected: 'rgba(0, 212, 255, 0.12)', // Subtle cyan selection
    disabled: 'rgba(255, 255, 255, 0.26)',
    disabledBackground: 'rgba(255, 255, 255, 0.12)',
  },
  
  // Status Colors
  success: {
    main: '#4CAF50',        // Green for success
    light: '#81C784',
    dark: '#2E7D32',
  },
  warning: {
    main: '#FF9800',        // Orange for warnings
    light: '#FFB74D',
    dark: '#F57C00',
  },
  error: {
    main: '#F44336',        // Red for errors
    light: '#E57373',
    dark: '#D32F2F',
  },
  info: {
    main: '#2196F3',        // Blue for info
    light: '#64B5F6',
    dark: '#1976D2',
  },
  
  // Professional borders and dividers
  divider: 'rgba(255, 255, 255, 0.12)',
  border: 'rgba(255, 255, 255, 0.23)',
};

// Typography for Professional Audio Software
const audioTypography = {
  fontFamily: [
    'Inter',                // Modern, highly readable
    'Roboto',              // Fallback
    'Helvetica Neue',      // Apple fallback
    'Arial',               // Universal fallback
    'sans-serif'
  ].join(','),
  
  // Professional font weights
  fontWeightLight: 300,
  fontWeightRegular: 400,
  fontWeightMedium: 500,
  fontWeightBold: 600,
  fontWeightExtraBold: 700,
  
  // Optimized font sizes for technical readability
  h1: {
    fontSize: '2.125rem',   // 34px - Main titles
    fontWeight: 600,
    lineHeight: 1.2,
    letterSpacing: '-0.02em',
  },
  h2: {
    fontSize: '1.75rem',    // 28px - Section headers
    fontWeight: 600,
    lineHeight: 1.3,
    letterSpacing: '-0.01em',
  },
  h3: {
    fontSize: '1.5rem',     // 24px - Subsection headers
    fontWeight: 500,
    lineHeight: 1.4,
  },
  h4: {
    fontSize: '1.25rem',    // 20px - Panel titles
    fontWeight: 500,
    lineHeight: 1.4,
  },
  h5: {
    fontSize: '1.125rem',   // 18px - Card titles
    fontWeight: 500,
    lineHeight: 1.4,
  },
  h6: {
    fontSize: '1rem',       // 16px - Widget titles
    fontWeight: 500,
    lineHeight: 1.4,
  },
  
  // Body text optimized for technical content
  body1: {
    fontSize: '0.875rem',   // 14px - Main body text
    fontWeight: 400,
    lineHeight: 1.5,
  },
  body2: {
    fontSize: '0.75rem',    // 12px - Secondary body text
    fontWeight: 400,
    lineHeight: 1.5,
  },
  
  // Specialized text for technical data
  caption: {
    fontSize: '0.6875rem',  // 11px - Captions and labels
    fontWeight: 400,
    lineHeight: 1.4,
    letterSpacing: '0.03em',
  },
  overline: {
    fontSize: '0.625rem',   // 10px - Overline text
    fontWeight: 500,
    lineHeight: 1.5,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
  },
  
  // Monospace for technical values
  mono: {
    fontFamily: ['SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', 'monospace'].join(','),
    fontSize: '0.875rem',
    fontWeight: 400,
    lineHeight: 1.4,
  },
};

// Professional component styling
const audioComponents = {
  // Card styling for professional panels
  MuiCard: {
    styleOverrides: {
      root: {
        backgroundColor: audioPalette.background.paper,
        border: `1px solid ${audioPalette.divider}`,
        borderRadius: 8,
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
        backdropFilter: 'blur(10px)',
      },
    },
  },
  
  // Button styling for professional controls
  MuiButton: {
    styleOverrides: {
      root: {
        borderRadius: 6,
        textTransform: 'none',
        fontWeight: 500,
        fontSize: '0.875rem',
        padding: '8px 16px',
        minHeight: 36,
      },
      contained: {
        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.3)',
        '&:hover': {
          boxShadow: '0 4px 8px rgba(0, 0, 0, 0.4)',
        },
      },
    },
  },
  
  // Slider styling for audio controls
  MuiSlider: {
    styleOverrides: {
      root: {
        height: 6,
        '& .MuiSlider-rail': {
          backgroundColor: audioPalette.background.elevated,
          opacity: 1,
        },
        '& .MuiSlider-track': {
          backgroundColor: audioPalette.primary.main,
          border: 'none',
          height: 6,
        },
        '& .MuiSlider-thumb': {
          backgroundColor: '#FFFFFF',
          border: `2px solid ${audioPalette.primary.main}`,
          width: 16,
          height: 16,
          '&:hover': {
            boxShadow: `0 0 0 8px rgba(0, 212, 255, 0.16)`,
          },
        },
      },
    },
  },
  
  // Input styling for precise value entry
  MuiTextField: {
    styleOverrides: {
      root: {
        '& .MuiOutlinedInput-root': {
          backgroundColor: audioPalette.background.surface,
          '& fieldset': {
            borderColor: audioPalette.border,
          },
          '&:hover fieldset': {
            borderColor: audioPalette.primary.light,
          },
          '&.Mui-focused fieldset': {
            borderColor: audioPalette.primary.main,
            borderWidth: 2,
          },
        },
      },
    },
  },
  
  // Tooltip styling for technical information
  MuiTooltip: {
    styleOverrides: {
      tooltip: {
        backgroundColor: audioPalette.background.elevated,
        border: `1px solid ${audioPalette.border}`,
        color: audioPalette.text.primary,
        fontSize: '0.75rem',
        fontWeight: 400,
        maxWidth: 300,
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.4)',
      },
      arrow: {
        color: audioPalette.background.elevated,
      },
    },
  },
  
  // Chip styling for tags and indicators
  MuiChip: {
    styleOverrides: {
      root: {
        backgroundColor: audioPalette.background.elevated,
        color: audioPalette.text.primary,
        fontWeight: 500,
        fontSize: '0.75rem',
        height: 24,
      },
      colorPrimary: {
        backgroundColor: audioPalette.primary.main,
        color: audioPalette.primary.contrastText,
      },
    },
  },
  
  // Tab styling for navigation
  MuiTabs: {
    styleOverrides: {
      root: {
        borderBottom: `1px solid ${audioPalette.divider}`,
        minHeight: 42,
      },
      indicator: {
        backgroundColor: audioPalette.primary.main,
        height: 3,
      },
    },
  },
  
  MuiTab: {
    styleOverrides: {
      root: {
        textTransform: 'none',
        fontWeight: 500,
        fontSize: '0.875rem',
        minHeight: 42,
        color: audioPalette.text.secondary,
        '&.Mui-selected': {
          color: audioPalette.primary.main,
        },
      },
    },
  },
};

// Create the professional audio theme
export const audioTheme = createTheme({
  palette: {
    mode: 'dark',
    ...audioPalette,
  },
  typography: audioTypography,
  components: audioComponents,
  
  // Custom spacing for precise layouts
  spacing: 8,
  
  // Professional border radius
  shape: {
    borderRadius: 6,
  },
  
  // Enhanced shadows for depth
  shadows: [
    'none',
    '0 1px 2px rgba(0, 0, 0, 0.2)',
    '0 2px 4px rgba(0, 0, 0, 0.25)',
    '0 4px 8px rgba(0, 0, 0, 0.3)',
    '0 6px 12px rgba(0, 0, 0, 0.35)',
    '0 8px 16px rgba(0, 0, 0, 0.4)',
    '0 12px 24px rgba(0, 0, 0, 0.45)',
    '0 16px 32px rgba(0, 0, 0, 0.5)',
    '0 20px 40px rgba(0, 0, 0, 0.55)',
    '0 24px 48px rgba(0, 0, 0, 0.6)',
    '0 32px 64px rgba(0, 0, 0, 0.65)',
    '0 40px 80px rgba(0, 0, 0, 0.7)',
    '0 48px 96px rgba(0, 0, 0, 0.75)',
    '0 56px 112px rgba(0, 0, 0, 0.8)',
    '0 64px 128px rgba(0, 0, 0, 0.85)',
    '0 72px 144px rgba(0, 0, 0, 0.9)',
    '0 80px 160px rgba(0, 0, 0, 0.95)',
    '0 88px 176px rgba(0, 0, 0, 1)',
    '0 96px 192px rgba(0, 0, 0, 1)',
    '0 104px 208px rgba(0, 0, 0, 1)',
    '0 112px 224px rgba(0, 0, 0, 1)',
    '0 120px 240px rgba(0, 0, 0, 1)',
    '0 128px 256px rgba(0, 0, 0, 1)',
    '0 136px 272px rgba(0, 0, 0, 1)',
    '0 144px 288px rgba(0, 0, 0, 1)',
  ],
  
  // Z-index hierarchy for professional layering
  zIndex: {
    mobileStepper: 1000,
    fab: 1050,
    speedDial: 1050,
    appBar: 1100,
    drawer: 1200,
    modal: 1300,
    snackbar: 1400,
    tooltip: 1500,
  },
});

// Export color utilities for direct use
export const audioColors = audioPalette;

// Export specialized color functions
export const getSplColor = (splValue: number): string => {
  if (splValue < 70) return audioPalette.audio.spl.low;
  if (splValue < 85) return audioPalette.audio.spl.moderate;
  if (splValue < 100) return audioPalette.audio.spl.high;
  return audioPalette.audio.spl.critical;
};

export const getFrequencyColor = (frequency: number): string => {
  if (frequency < 60) return audioPalette.audio.frequency.sub;
  if (frequency < 250) return audioPalette.audio.frequency.bass;
  if (frequency < 4000) return audioPalette.audio.frequency.mid;
  if (frequency < 16000) return audioPalette.audio.frequency.high;
  return audioPalette.audio.frequency.ultra;
};