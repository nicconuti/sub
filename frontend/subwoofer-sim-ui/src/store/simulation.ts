/**
 * Simulation State Management
 * Zustand store for simulation data and real-time updates
 */

import { create } from 'zustand';
import { devtools, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

// Temporary type definitions until import issue is resolved
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

interface SimulationParams {
  frequency: number;
  speed_of_sound: number;
  grid_resolution: number;
  room_vertices: [number, number][];
  target_areas: any[];
  avoidance_areas: any[];
}

interface SplMapData {
  X: number[][];
  Y: number[][];
  SPL: number[][];
  frequency: number;
  timestamp: number;
  statistics: any;
}

interface Project {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  sources: SourceData[];
  simulation_params: SimulationParams;
  target_areas: any[];
  avoidance_areas: any[];
  results?: any;
  metadata: any;
}

interface OptimizationResults {
  final_configuration: SourceData[];
  convergence_data: number[];
  total_generations: number;
  final_fitness: number;
  target_achieved: boolean;
  computation_time: number;
}

interface SimulationProgress {
  progress: number;           // 0-100
  current_step: string;
  estimated_time?: number;
  chunks_received?: number;
  total_chunks?: number;
}

interface SimulationState {
  // Project Management
  currentProject: Project | null;
  projects: Project[];
  projectsLoading: 'idle' | 'loading' | 'success' | 'error';
  
  // Simulation Data
  splMap: SplMapData | null;
  splChunks: Record<number, any>; // For progressive loading
  simulationProgress: SimulationProgress;
  isSimulating: boolean;
  lastSimulationTime: number | null;
  
  // Sources Management
  selectedSources: string[];
  hoveredSource: string | null;
  
  // Optimization
  optimizationResults: OptimizationResults | null;
  isOptimizing: boolean;
  optimizationProgress: {
    generation: number;
    total_generations: number;
    best_fitness: number;
    convergence_rate: number;
  } | null;
  
  // Parameters
  simulationParams: SimulationParams;
  
  // Actions
  setCurrentProject: (project: Project | null) => void;
  updateProject: (updates: Partial<Project>) => void;
  createDefaultProject: () => void;
  addSource: (source: SourceData) => void;
  updateSource: (sourceId: string, updates: Partial<SourceData>) => void;
  removeSource: (sourceId: string) => void;
  selectSources: (sourceIds: string[]) => void;
  setHoveredSource: (sourceId: string | null) => void;
  
  // Simulation Actions
  startSimulation: () => Promise<void>;
  stopSimulation: () => Promise<void>;
  updateSimulationProgress: (progress: SimulationProgress) => void;
  addSplChunk: (chunkIndex: number, chunk: any) => void;
  completeSplMap: (splMap: SplMapData) => void;
  
  // Optimization Actions
  startOptimization: (config: any) => void;
  stopOptimization: () => void;
  updateOptimizationProgress: (progress: any) => void;
  completeOptimization: (results: OptimizationResults) => void;
  
  // Parameters Actions
  updateSimulationParams: (params: Partial<SimulationParams>) => void;
  
  // Utility Actions
  resetSimulation: () => void;
  clearResults: () => void;
}

// Default simulation parameters
const defaultSimulationParams: SimulationParams = {
  frequency: 80,
  speed_of_sound: 343,
  grid_resolution: 0.1,
  room_vertices: [[0, 0], [10, 0], [10, 8], [0, 8]],
  target_areas: [],
  avoidance_areas: [],
};

export const useSimulationStore = create<SimulationState>()(
  devtools(
    subscribeWithSelector(
      immer((set, get) => ({
        // Initial State
        currentProject: null,
        projects: [],
        projectsLoading: 'idle',
        
        splMap: null,
        splChunks: {},
        simulationProgress: {
          progress: 0,
          current_step: 'idle',
        },
        isSimulating: false,
        lastSimulationTime: null,
        
        selectedSources: [],
        hoveredSource: null,
        
        optimizationResults: null,
        isOptimizing: false,
        optimizationProgress: null,
        
        simulationParams: defaultSimulationParams,
        
        // Project Actions
        setCurrentProject: (project) => set((state) => {
          state.currentProject = project;
          if (project) {
            state.simulationParams = { ...project.simulation_params };
          }
        }),
        
        createDefaultProject: () => set((state) => {
          const defaultProject: Project = {
            id: `project_${Date.now()}`,
            name: 'Test Project',
            description: 'Default test project for simulation',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            simulation_params: {
              frequency: 80,
              grid_resolution: 0.2,
              room_dimensions: [10, 8, 3],
              room_vertices: [
                [0, 0], [10, 0], [10, 8], [0, 8]
              ],
              calculation_bounds: {
                x_min: 0, x_max: 10,
                y_min: 0, y_max: 8,
              },
              environment: {
                temperature: 20,
                humidity: 50,
                pressure: 101325,
              },
            },
            sources: [
              {
                id: 'source_1',
                x: 2,
                y: 4,
                spl_rms: 105,
                gain_db: 0,
                delay_ms: 0,
                angle: 0,
                polarity: 1,
                name: 'Subwoofer 1',
                enabled: true,
                color: '#00D4FF',
              },
              {
                id: 'source_2', 
                x: 8,
                y: 4,
                spl_rms: 105,
                gain_db: 0,
                delay_ms: 0,
                angle: 0,
                polarity: 1,
                name: 'Subwoofer 2',
                enabled: true,
                color: '#FF6B35',
              },
            ],
          };
          
          state.currentProject = defaultProject;
          state.simulationParams = { ...defaultProject.simulation_params };
        }),
        
        updateProject: (updates) => set((state) => {
          if (state.currentProject) {
            Object.assign(state.currentProject, updates);
            state.currentProject.updated_at = new Date().toISOString();
          }
        }),
        
        addSource: (source) => set((state) => {
          if (state.currentProject) {
            state.currentProject.sources.push(source);
            state.currentProject.updated_at = new Date().toISOString();
          }
        }),
        
        updateSource: (sourceId, updates) => set((state) => {
          if (state.currentProject) {
            const sourceIndex = state.currentProject.sources.findIndex(s => s.id === sourceId);
            if (sourceIndex !== -1) {
              Object.assign(state.currentProject.sources[sourceIndex], updates);
              state.currentProject.updated_at = new Date().toISOString();
            }
          }
        }),
        
        removeSource: (sourceId) => set((state) => {
          if (state.currentProject) {
            state.currentProject.sources = state.currentProject.sources.filter(s => s.id !== sourceId);
            state.selectedSources = state.selectedSources.filter(id => id !== sourceId);
            state.currentProject.updated_at = new Date().toISOString();
          }
        }),
        
        selectSources: (sourceIds) => set((state) => {
          state.selectedSources = sourceIds;
        }),
        
        setHoveredSource: (sourceId) => set((state) => {
          state.hoveredSource = sourceId;
        }),
        
        // Simulation Actions
        startSimulation: async () => {
          let state = get();
          
          // Check if we have a project and connection
          if (!state.currentProject) {
            console.log('📁 No current project, creating default project...');
            // Create a default project with test sources
            get().createDefaultProject();
            // Get updated state after creating project
            state = get();
          }
          
          // Import connection store dynamically to avoid circular imports
          const { useConnectionStore } = await import('./connection');
          const connectionState = useConnectionStore.getState();
          
          if (!connectionState.is_connected) {
            console.error('❌ Not connected to backend');
            return;
          }
          
          if (!state.currentProject) {
            console.error('❌ Failed to create default project');
            return;
          }
          
          try {
            // Update local state first
            set((state) => {
              state.isSimulating = true;
              state.simulationProgress = {
                progress: 0,
                current_step: 'initializing',
              };
              state.splChunks = {};
              state.splMap = null;
            });
            
            // Convert frontend params to backend format
            const backendParams = {
              frequency: state.currentProject.simulation_params.frequency,
              speed_of_sound: 343.0, // Default speed of sound
              grid_resolution: state.currentProject.simulation_params.grid_resolution,
              room_vertices: state.currentProject.simulation_params.room_vertices,
              target_areas: [],
              avoidance_areas: [],
            };
            
            // Convert sources to backend format
            const backendSources = state.currentProject.sources.map(source => ({
              id: source.id,
              x: source.x,
              y: source.y,
              spl_rms: source.spl_rms,
              gain_db: source.gain_db,
              delay_ms: source.delay_ms,
              angle: source.angle,
              polarity: source.polarity,
            }));
            
            // Send start command to backend (backend expects 'params' not 'simulation_params')
            await connectionState.sendMessage('simulation:start', {
              params: backendParams,
              sources: backendSources,
            });
            
            console.log('🚀 Simulation start command sent');
            
          } catch (error) {
            console.error('❌ Failed to start simulation:', error);
            // Revert state on error
            set((state) => {
              state.isSimulating = false;
              state.simulationProgress = {
                progress: 0,
                current_step: 'error',
              };
            });
          }
        },
        
        stopSimulation: async () => {
          try {
            // Import connection store dynamically
            const { useConnectionStore } = await import('./connection');
            const connectionState = useConnectionStore.getState();
            
            if (connectionState.is_connected) {
              await connectionState.sendMessage('simulation:stop', {});
              console.log('🛑 Simulation stop command sent');
            }
            
            // Update local state
            set((state) => {
              state.isSimulating = false;
              state.simulationProgress = {
                progress: 0,
                current_step: 'stopped',
              };
            });
            
          } catch (error) {
            console.error('❌ Failed to stop simulation:', error);
            // Still update local state even if backend call fails
            set((state) => {
              state.isSimulating = false;
              state.simulationProgress = {
                progress: 0,
                current_step: 'stopped',
              };
            });
          }
        },
        
        updateSimulationProgress: (progress) => set((state) => {
          state.simulationProgress = progress;
        }),
        
        addSplChunk: (chunkIndex, chunk) => set((state) => {
          state.splChunks[chunkIndex] = chunk;
          
          // Update progress based on chunks received
          if (chunk.total_chunks) {
            const chunksReceived = Object.keys(state.splChunks).length;
            const progressPercent = (chunksReceived / chunk.total_chunks) * 100;
            state.simulationProgress.progress = Math.min(95, 30 + progressPercent * 0.6);
            state.simulationProgress.chunks_received = chunksReceived;
            state.simulationProgress.total_chunks = chunk.total_chunks;
          }
        }),
        
        completeSplMap: (splMap) => set((state) => {
          state.splMap = splMap;
          state.isSimulating = false;
          state.lastSimulationTime = Date.now();
          state.simulationProgress = {
            progress: 100,
            current_step: 'complete',
          };
          
          // Store results in project
          if (state.currentProject) {
            if (!state.currentProject.results) {
              state.currentProject.results = {
                spl_maps: {},
                last_calculated: new Date().toISOString(),
              };
            }
            const frequencyKey = (splMap.frequency || 80).toString();
            state.currentProject.results.spl_maps[frequencyKey] = splMap;
            state.currentProject.results.last_calculated = new Date().toISOString();
            state.currentProject.updated_at = new Date().toISOString();
          }
        }),
        
        // Optimization Actions
        startOptimization: (config) => set((state) => {
          state.isOptimizing = true;
          state.optimizationProgress = {
            generation: 0,
            total_generations: config.max_generations,
            best_fitness: 0,
            convergence_rate: 0,
          };
          state.optimizationResults = null;
        }),
        
        stopOptimization: () => set((state) => {
          state.isOptimizing = false;
          state.optimizationProgress = null;
        }),
        
        updateOptimizationProgress: (progress) => set((state) => {
          state.optimizationProgress = progress;
        }),
        
        completeOptimization: (results) => set((state) => {
          state.optimizationResults = results;
          state.isOptimizing = false;
          state.optimizationProgress = null;
          
          // Store results in project
          if (state.currentProject) {
            if (!state.currentProject.results) {
              state.currentProject.results = {
                spl_maps: {},
                last_calculated: new Date().toISOString(),
              };
            }
            state.currentProject.results.optimization_results = results;
            state.currentProject.updated_at = new Date().toISOString();
          }
        }),
        
        // Parameters Actions
        updateSimulationParams: (params) => set((state) => {
          Object.assign(state.simulationParams, params);
          
          // Update project if exists
          if (state.currentProject) {
            Object.assign(state.currentProject.simulation_params, params);
            state.currentProject.updated_at = new Date().toISOString();
          }
        }),
        
        // Utility Actions
        resetSimulation: () => set((state) => {
          state.isSimulating = false;
          state.simulationProgress = {
            progress: 0,
            current_step: 'idle',
          };
          state.splMap = null;
          state.splChunks = {};
          state.lastSimulationTime = null;
        }),
        
        clearResults: () => set((state) => {
          state.splMap = null;
          state.splChunks = {};
          state.optimizationResults = null;
          state.lastSimulationTime = null;
          
          if (state.currentProject && state.currentProject.results) {
            state.currentProject.results.spl_maps = {};
            state.currentProject.results.optimization_results = undefined;
          }
        }),
      }))
    ),
    {
      name: 'simulation-store',
      serialize: {
        // Don't serialize large data structures to avoid localStorage issues
        options: {
          map: (key, value) => {
            if (key === 'splChunks') return {}; // Don't persist chunks
            if (key === 'splMap' && value && JSON.stringify(value).length > 100000) {
              return null; // Don't persist very large SPL maps
            }
            return value;
          }
        }
      }
    }
  )
);

// Selectors for derived state
export const useCurrentSources = () => 
  useSimulationStore(state => state.currentProject?.sources || []);

export const useSelectedSources = () =>
  useSimulationStore(state => {
    const sources = state.currentProject?.sources || [];
    return sources.filter(source => state.selectedSources.includes(source.id));
  });

export const useSimulationStatus = () =>
  useSimulationStore(state => ({
    isSimulating: state.isSimulating,
    progress: state.simulationProgress,
    hasResults: !!state.splMap,
    lastCalculated: state.lastSimulationTime,
  }));

export const useOptimizationStatus = () =>
  useSimulationStore(state => ({
    isOptimizing: state.isOptimizing,
    progress: state.optimizationProgress,
    results: state.optimizationResults,
  }));

// Action creators for WebSocket integration
export const simulationActions = {
  // Called when WebSocket events are received
  onSimulationProgress: (progress: SimulationProgress) => {
    useSimulationStore.getState().updateSimulationProgress(progress);
  },
  
  onSplChunk: (chunkData: any) => {
    useSimulationStore.getState().addSplChunk(chunkData.chunk_index, chunkData);
  },
  
  onSimulationComplete: (completionData: any) => {
    const state = useSimulationStore.getState();
    
    // Reconstruct SPL map from chunks
    if (Object.keys(state.splChunks).length > 0) {
      // Get chunks and sort by index
      const sortedChunks = Object.entries(state.splChunks)
        .sort(([a], [b]) => parseInt(a) - parseInt(b))
        .map(([_, chunk]) => chunk);
      
      if (sortedChunks.length > 0) {
        // Reconstruct the complete SPL map from chunks
        const firstChunk = sortedChunks[0];
        const totalChunks = firstChunk.total_chunks || sortedChunks.length;
        
        // Combine all X, Y, and SPL data properly
        console.log('🔧 Reconstructing SPL map from', sortedChunks.length, 'chunks');
        
        const combinedX: number[][] = [];
        const combinedY: number[][] = [];
        const combinedSPL: number[][] = [];
        
        sortedChunks.forEach((chunk, index) => {
          if (chunk.X && chunk.Y && chunk.SPL && 
              Array.isArray(chunk.X) && Array.isArray(chunk.Y) && Array.isArray(chunk.SPL)) {
            
            console.log(`📊 Chunk ${index}: ${chunk.X.length} rows`);
            
            // Each chunk contains a 2D array - concatenate the rows
            // chunk.X is [[x1,x2,x3], [x4,x5,x6], ...] 
            // We want to combine all chunks into one big 2D array
            chunk.X.forEach(row => combinedX.push(row));
            chunk.Y.forEach(row => combinedY.push(row));
            chunk.SPL.forEach(row => combinedSPL.push(row));
          }
        });
        
        console.log('✅ Combined data:', {
          X: combinedX.length,
          Y: combinedY.length, 
          SPL: combinedSPL.length
        });
        
        // Create complete SPL map
        const completeSplMap: SplMapData = {
          X: combinedX,
          Y: combinedY,
          SPL: combinedSPL,
          frequency: completionData.statistics?.frequency || 80, // Use frequency from completion data
          timestamp: Date.now(),
          statistics: completionData.statistics || {}
        };
        
        useSimulationStore.getState().completeSplMap(completeSplMap);
      }
    }
    
    // Update simulation state to completed
    useSimulationStore.setState((state) => ({
      ...state,
      isSimulating: false,
      simulationProgress: {
        progress: 100,
        current_step: 'complete'
      }
    }));
  },
  
  onOptimizationGeneration: (generationData: any) => {
    useSimulationStore.getState().updateOptimizationProgress(generationData);
  },
  
  onOptimizationComplete: (results: OptimizationResults) => {
    useSimulationStore.getState().completeOptimization(results);
  },
  
  onParameterValidation: (validation: any) => {
    // Handle parameter validation feedback
    console.log('Parameter validation:', validation);
  },
  
  onSourceMoved: (data: { sourceId: string; x: number; y: number }) => {
    useSimulationStore.getState().updateSource(data.sourceId, { x: data.x, y: data.y });
  },
  
  onError: (error: any) => {
    console.error('Simulation error:', error);
    // Could update error state here
  },
};