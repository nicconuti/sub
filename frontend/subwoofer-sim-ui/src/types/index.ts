/**
 * TypeScript Type Definitions for Subwoofer Simulation
 * Professional audio software types with comprehensive validation
 */

// ========================
// CORE SIMULATION TYPES
// ========================

export interface SourceData {
  id: string;
  x: number;                    // Position X in meters
  y: number;                    // Position Y in meters
  spl_rms: number;             // SPL RMS in dB (50-150)
  gain_db: number;             // Gain in dB (-60 to +20)
  delay_ms: number;            // Delay in milliseconds (0-1000)
  angle: number;               // Rotation angle in degrees (0-360)
  polarity: -1 | 1;            // Polarity: -1 (inverted) or 1 (normal)
  name?: string;               // Optional display name
  color?: string;              // Optional color for visualization
  enabled?: boolean;           // Whether source is active
}

export interface SimulationParams {
  frequency: number;           // Frequency in Hz (20-20000)
  speed_of_sound: number;      // Speed of sound in m/s (300-400)
  grid_resolution: number;     // Grid resolution in meters (0.01-1.0)
  room_vertices: [number, number][]; // Room boundary vertices
  target_areas: TargetArea[];  // Areas to optimize for
  avoidance_areas: AvoidanceArea[]; // Areas to avoid
}

export interface TargetArea {
  id: string;
  vertices: [number, number][];
  target_spl: number;          // Target SPL in dB
  tolerance: number;           // Tolerance in dB
  weight: number;              // Optimization weight (0-1)
  name: string;
  color?: string;
}

export interface AvoidanceArea {
  id: string;
  vertices: [number, number][];
  max_spl: number;             // Maximum allowed SPL in dB
  name: string;
  color?: string;
}

// ========================
// SPL DATA TYPES
// ========================

export interface SplMapData {
  X: number[][];               // X coordinate grid
  Y: number[][];               // Y coordinate grid  
  SPL: number[][];             // SPL values grid
  frequency: number;           // Frequency for this map
  timestamp: number;           // When calculated
  statistics: SplStatistics;   // Statistical data
}

export interface SplChunkData {
  X: number[][];
  Y: number[][];
  SPL: number[][];
  chunk_index: number;
  total_chunks: number;
  is_last: boolean;
}

export interface SplStatistics {
  min_spl: number;
  max_spl: number;
  mean_spl: number;
  std_dev: number;
  coverage_85db: number;       // Percentage above 85dB
  coverage_100db: number;      // Percentage above 100dB
}

export interface SplPreviewData {
  point: [number, number];     // [x, y] coordinates
  spl_value: number;          // SPL at point
  frequency: number;          // Frequency
  timestamp: number;          // When calculated
}

// ========================
// PROJECT TYPES
// ========================

export interface Project {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  sources: SourceData[];
  simulation_params: SimulationParams;
  target_areas: TargetArea[];
  avoidance_areas: AvoidanceArea[];
  results?: ProjectResults;
  metadata: ProjectMetadata;
}

export interface ProjectResults {
  spl_maps: { [frequency: string]: SplMapData };
  optimization_results?: OptimizationResults;
  last_calculated: string;
}

export interface ProjectMetadata {
  author?: string;
  version: string;
  tags: string[];
  notes: string;
  export_settings?: ExportSettings;
}

export interface ExportSettings {
  format: 'json' | 'xlsx' | 'csv';
  include_spl_maps: boolean;
  include_optimization: boolean;
  colormap: string;
}

// ========================
// OPTIMIZATION TYPES  
// ========================

export interface OptimizationConfig {
  population_size: number;     // GA population size (10-200)
  max_generations: number;     // Maximum generations (10-1000)
  mutation_rate: number;       // Mutation rate (0.001-0.1)
  crossover_rate: number;      // Crossover rate (0.1-1.0)
  target_spl: number;         // Target SPL in dB (60-120)
  tolerance: number;          // Tolerance in dB (1-10)
}

export interface OptimizationGeneration {
  generation: number;
  best_fitness: number;
  avg_fitness: number;
  best_configuration: SourceData[];
  convergence_rate: number;
  estimated_time_remaining?: number;
}

export interface OptimizationResults {
  final_configuration: SourceData[];
  convergence_data: number[];
  total_generations: number;
  final_fitness: number;
  target_achieved: boolean;
  computation_time: number;
}

// ========================
// WEBSOCKET EVENT TYPES
// ========================

export type ClientEventType = 
  | 'project:load'
  | 'project:save'
  | 'project:join'
  | 'simulation:start'
  | 'simulation:stop'
  | 'simulation:parameter_update'
  | 'optimization:start'
  | 'optimization:stop'
  | 'source:move'
  | 'source:rotate'
  | 'room:vertex_move'
  | 'spl_preview:request';

export type ServerEventType =
  | 'connected'
  | 'error'
  | 'simulation:progress'
  | 'simulation:spl_chunk'
  | 'simulation:complete'
  | 'optimization:generation'
  | 'optimization:complete'
  | 'parameter:validation'
  | 'ui:source_moved'
  | 'ui:room_vertex_moved'
  | 'ui:spl_preview';

export interface BaseEvent<T = any> {
  type: ClientEventType | ServerEventType;
  data: T;
  timestamp: number;
  session_id?: string;
  request_id?: string;
}

export interface ClientEvent<T = any> extends BaseEvent<T> {
  type: ClientEventType;
}

export interface ServerEvent<T = any> extends BaseEvent<T> {
  type: ServerEventType;
}

// ========================
// UI STATE TYPES
// ========================

export interface ViewState {
  zoom: number;               // Zoom level (0.1-10)
  pan: [number, number];      // Pan offset [x, y]
  center: [number, number];   // View center [x, y]
  rotation: number;           // View rotation in degrees
}

export interface SelectionState {
  selected_sources: string[]; // Selected source IDs
  selected_areas: string[];   // Selected area IDs
  hover_source?: string;      // Hovered source ID
  hover_point?: [number, number]; // Hovered point
}

export interface UiState {
  sidebar_collapsed: boolean;
  active_tab: string;
  view_state: ViewState;
  selection_state: SelectionState;
  is_simulating: boolean;
  is_optimizing: boolean;
  show_grid: boolean;
  show_spl_overlay: boolean;
  colormap: string;
  spl_range: [number, number];
}

// ========================
// CONNECTION TYPES
// ========================

export interface ConnectionState {
  is_connected: boolean;
  connection_error?: string;
  session_id?: string;
  server_version?: string;
  last_ping: number;
  reconnect_attempts: number;
}

export interface WebSocketState {
  connection: ConnectionState;
  pending_requests: Map<string, PendingRequest>;
  message_queue: ServerEvent[];
  statistics: ConnectionStatistics;
}

export interface PendingRequest {
  request_id: string;
  type: ClientEventType;
  timestamp: number;
  timeout: number;
  resolve: (data: any) => void;
  reject: (error: Error) => void;
}

export interface ConnectionStatistics {
  messages_sent: number;
  messages_received: number;
  bytes_sent: number;
  bytes_received: number;
  connection_uptime: number;
  last_error?: string;
}

// ========================
// VALIDATION TYPES
// ========================

export interface ValidationResult {
  valid: boolean;
  message?: string;
  suggested_value?: any;
  constraints?: ValidationConstraints;
}

export interface ValidationConstraints {
  min?: number;
  max?: number;
  step?: number;
  options?: any[];
  pattern?: string;
  required?: boolean;
}

export interface ParameterValidation {
  parameter: string;
  valid: boolean;
  message?: string;
  suggested_value?: any;
  constraints?: ValidationConstraints;
}

// ========================
// PERFORMANCE TYPES
// ========================

export interface PerformanceMetrics {
  render_time: number;        // Component render time in ms
  calculation_time: number;   // Calculation time in ms
  websocket_latency: number;  // WebSocket round-trip time in ms
  memory_usage: number;       // Memory usage in MB
  fps: number;               // Frames per second
}

export interface SystemInfo {
  user_agent: string;
  screen_resolution: [number, number];
  color_depth: number;
  memory_limit: number;
  cpu_cores: number;
  gpu_vendor?: string;
}

// ========================
// UTILITY TYPES
// ========================

export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

export interface AsyncState<T> {
  data?: T;
  loading: boolean;
  error?: string;
  last_updated?: number;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: number;
}

// ========================
// CONSTANTS
// ========================

export const SIMULATION_LIMITS = {
  FREQUENCY: { MIN: 20, MAX: 20000 },
  SPL_RMS: { MIN: 50, MAX: 150 },
  GAIN_DB: { MIN: -60, MAX: 20 },
  DELAY_MS: { MIN: 0, MAX: 1000 },
  ANGLE: { MIN: 0, MAX: 360 },
  GRID_RESOLUTION: { MIN: 0.01, MAX: 1.0 },
  SPEED_OF_SOUND: { MIN: 300, MAX: 400 },
  MAX_SOURCES: 100,
  MAX_ROOM_VERTICES: 50,
} as const;

export const UI_CONSTANTS = {
  SIDEBAR_WIDTH: 320,
  HEADER_HEIGHT: 64,
  TAB_HEIGHT: 48,
  MIN_ZOOM: 0.1,
  MAX_ZOOM: 10,
  GRID_SNAP: 0.1,
  SELECTION_TOLERANCE: 10,
} as const;

export const WEBSOCKET_CONFIG = {
  RECONNECT_INTERVAL: 3000,
  MAX_RECONNECT_ATTEMPTS: 10,
  PING_INTERVAL: 30000,
  MESSAGE_TIMEOUT: 10000,
  CHUNK_SIZE: 1000,
} as const;

// ========================
// TYPE GUARDS
// ========================

export const isSourceData = (obj: any): obj is SourceData => {
  return (
    typeof obj === 'object' &&
    typeof obj.id === 'string' &&
    typeof obj.x === 'number' &&
    typeof obj.y === 'number' &&
    typeof obj.spl_rms === 'number' &&
    typeof obj.gain_db === 'number' &&
    typeof obj.delay_ms === 'number' &&
    typeof obj.angle === 'number' &&
    (obj.polarity === -1 || obj.polarity === 1)
  );
};

export const isServerEvent = (obj: any): obj is ServerEvent => {
  return (
    typeof obj === 'object' &&
    typeof obj.type === 'string' &&
    typeof obj.timestamp === 'number' &&
    obj.data !== undefined
  );
};

export const isSplMapData = (obj: any): obj is SplMapData => {
  return (
    typeof obj === 'object' &&
    Array.isArray(obj.X) &&
    Array.isArray(obj.Y) &&
    Array.isArray(obj.SPL) &&
    typeof obj.frequency === 'number' &&
    typeof obj.timestamp === 'number'
  );
};