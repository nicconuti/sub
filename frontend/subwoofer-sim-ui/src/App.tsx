/**
 * Professional Subwoofer Simulation Application
 * Main React application with professional audio software design
 */

import React, { useEffect } from 'react';
import { ThemeProvider, CssBaseline, GlobalStyles } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import { audioTheme } from './theme/audioTheme';
import { AppLayout } from './components/layout/AppLayout';
import { SplViewer } from './components/visualization/SplViewer';
import { useConnectionStore } from './store/connection';
import { useSimulationStore } from './store/simulation';
import { simulationActions } from './store/simulation';

// Create React Query client for API management
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

// Global styles for professional audio application
const globalStyles = (
  <GlobalStyles
    styles={(theme) => ({
      '*': {
        boxSizing: 'border-box',
      },
      html: {
        height: '100%',
        // Prevent text selection on UI elements
        WebkitUserSelect: 'none',
        MozUserSelect: 'none',
        msUserSelect: 'none',
        userSelect: 'none',
        // Disable text highlighting
        WebkitTouchCallout: 'none',
        WebkitTapHighlightColor: 'transparent',
      },
      body: {
        height: '100%',
        margin: 0,
        padding: 0,
        fontFamily: theme.typography.fontFamily,
        backgroundColor: theme.palette.background.default,
        color: theme.palette.text.primary,
        // Professional anti-aliasing
        WebkitFontSmoothing: 'antialiased',
        MozOsxFontSmoothing: 'grayscale',
        // Prevent overscroll
        overscrollBehavior: 'none',
      },
      '#root': {
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
      },
      // Professional scrollbar styling
      '::-webkit-scrollbar': {
        width: 8,
        height: 8,
      },
      '::-webkit-scrollbar-track': {
        backgroundColor: 'transparent',
      },
      '::-webkit-scrollbar-thumb': {
        backgroundColor: theme.palette.divider,
        borderRadius: 4,
        border: `2px solid ${theme.palette.background.default}`,
        '&:hover': {
          backgroundColor: theme.palette.text.disabled,
        },
      },
      '::-webkit-scrollbar-corner': {
        backgroundColor: 'transparent',
      },
      // Selection styling
      '::selection': {
        backgroundColor: theme.palette.primary.main,
        color: theme.palette.primary.contrastText,
      },
      '::-moz-selection': {
        backgroundColor: theme.palette.primary.main,
        color: theme.palette.primary.contrastText,
      },
      // Focus outline styling for accessibility
      '*:focus-visible': {
        outline: `2px solid ${theme.palette.primary.main}`,
        outlineOffset: 2,
        borderRadius: 2,
      },
      // Disable outline for mouse users
      '*:focus:not(:focus-visible)': {
        outline: 'none',
      },
      // Professional input styling
      'input, textarea, select': {
        WebkitUserSelect: 'text',
        MozUserSelect: 'text',
        msUserSelect: 'text',
        userSelect: 'text',
      },
      // Performance optimizations
      '.canvas-container': {
        willChange: 'transform',
        backfaceVisibility: 'hidden',
        transform: 'translateZ(0)',
      },
      // Professional animations
      '.fade-enter': {
        opacity: 0,
      },
      '.fade-enter-active': {
        opacity: 1,
        transition: 'opacity 200ms ease-in',
      },
      '.fade-exit': {
        opacity: 1,
      },
      '.fade-exit-active': {
        opacity: 0,
        transition: 'opacity 200ms ease-out',
      },
    })}
  />
);

const App: React.FC = () => {
  const { connect, socket, is_connected, message_queue } = useConnectionStore();
  const { splMap, currentProject } = useSimulationStore();

  // Initialize WebSocket connection on app start
  useEffect(() => {
    const initializeConnection = async () => {
      try {
        await connect('ws://localhost:8001/ws');
        console.log('🚀 WebSocket connection initialized');
      } catch (error) {
        console.error('❌ Failed to initialize WebSocket connection:', error);
      }
    };

    if (!socket && !is_connected) {
      initializeConnection();
    }
  }, [connect, socket, is_connected]);

  // Process incoming WebSocket messages
  useEffect(() => {
    if (message_queue.length === 0) return;

    // Process each message in the queue
    message_queue.forEach((message) => {
      switch (message.type) {
        case 'simulation:progress':
          simulationActions.onSimulationProgress(message.data);
          break;
        
        case 'simulation:spl_chunk':
          simulationActions.onSplChunk(message.data);
          break;
        
        case 'simulation:complete':
          simulationActions.onSimulationComplete(message.data);
          break;
        
        case 'optimization:generation':
          simulationActions.onOptimizationGeneration(message.data);
          break;
        
        case 'optimization:complete':
          simulationActions.onOptimizationComplete(message.data);
          break;
        
        case 'parameter:validation':
          simulationActions.onParameterValidation(message.data);
          break;
        
        case 'ui:source_moved':
          simulationActions.onSourceMoved(message.data);
          break;
        
        case 'error':
          simulationActions.onError(message.data);
          break;
        
        default:
          console.log('📡 Unhandled WebSocket message:', message.type, message.data);
          break;
      }
    });

    // Clear processed messages
    useConnectionStore.getState().updateStatistics({
      messages_received: useConnectionStore.getState().statistics.messages_received + message_queue.length,
    });
    
    // Clear message queue after processing
    useConnectionStore.setState((state) => ({ ...state, message_queue: [] }));
  }, [message_queue]);

  // View state management for SPL viewer
  const [viewState, setViewState] = React.useState({
    zoom: 1,
    pan: [0, 0] as [number, number],
    center: [5, 4] as [number, number],
    rotation: 0,
  });

  const handleSourceSelect = (sourceId: string) => {
    const currentSelection = useSimulationStore.getState().selectedSources;
    const newSelection = currentSelection.includes(sourceId)
      ? currentSelection.filter(id => id !== sourceId)
      : [...currentSelection, sourceId];
    
    useSimulationStore.getState().selectSources(newSelection);
  };

  const handlePointHover = (point: [number, number]) => {
    // Could trigger SPL preview request here
    console.log('Point hover:', point);
  };

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={audioTheme}>
        <CssBaseline />
        {globalStyles}
        
        <AppLayout>
          <div className="canvas-container" style={{ width: '100%', height: '100%' }}>
            <SplViewer
              splData={splMap || undefined}
              sources={currentProject?.sources || []}
              viewState={viewState}
              onViewStateChange={setViewState}
              onSourceSelect={handleSourceSelect}
              onPointHover={handlePointHover}
              showSources={true}
              showGrid={true}
              interactive={true}
              colorscale="acoustic"
            />
          </div>
        </AppLayout>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

export default App;
