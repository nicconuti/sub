/**
 * WebSocket Connection State Management
 * Manages real-time connection to simulation backend
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

// Temporary type definitions until import issue is resolved
interface ConnectionState {
  is_connected: boolean;
  connection_error?: string | null;
  session_id?: string;
  server_version?: string;
  last_ping: number;
  reconnect_attempts: number;
}

interface ClientEvent<T = any> {
  type: string;
  data: T;
  timestamp: number;
  session_id?: string;
  request_id?: string;
}

interface ServerEvent<T = any> {
  type: string;
  data: T;
  timestamp: number;
  session_id?: string;
  request_id?: string;
}

interface PendingRequest {
  request_id: string;
  type: string;
  timestamp: number;
  timeout: any;
  resolve: (data: any) => void;
  reject: (error: Error) => void;
}

interface ConnectionStatistics {
  messages_sent: number;
  messages_received: number;
  bytes_sent: number;
  bytes_received: number;
  connection_uptime: number;
  last_error?: string;
}

// Extended state interface
interface ConnectionStore extends ConnectionState {
  // WebSocket instance
  socket: WebSocket | null;
  
  // Connection management
  pending_requests: Record<string, PendingRequest>;
  message_queue: ServerEvent[];
  statistics: ConnectionStatistics;
  auto_reconnect: boolean;
  
  // Actions
  connect: (url?: string) => Promise<void>;
  disconnect: () => void;
  send: <T = any>(event: ClientEvent<T>) => Promise<any>;
  sendMessage: <T = any>(type: string, data: T) => Promise<any>;
  ping: () => Promise<boolean>;
  addToMessageQueue: (event: ServerEvent) => void;
  updateStatistics: (updates: Partial<ConnectionStatistics>) => void;
  clearMessageQueue: () => void;
}

// Initial statistics
const initialStatistics: ConnectionStatistics = {
  messages_sent: 0,
  messages_received: 0,
  bytes_sent: 0,
  bytes_received: 0,
  connection_uptime: 0,
};

export const useConnectionStore = create<ConnectionStore>()(
  devtools(
    immer((set, get) => ({
      // Initial state
      is_connected: false,
      connection_error: null,
      session_id: undefined,
      server_version: undefined,
      last_ping: 0,
      reconnect_attempts: 0,
      socket: null,
      pending_requests: {},
      message_queue: [],
      statistics: { ...initialStatistics },
      auto_reconnect: true,
      
      // Connection management
      connect: async (url = 'ws://localhost:8001/ws') => {
        const state = get();
        
        // Don't connect if already connected
        if (state.is_connected && state.socket) {
          return;
        }
        
        try {
          const socket = new WebSocket(url);
          
          // Connection event handlers
          socket.onopen = () => {
            console.log('🔗 WebSocket connected');
            set((state) => {
              state.is_connected = true;
              state.connection_error = null;
              state.reconnect_attempts = 0;
              state.last_ping = Date.now();
              state.statistics.connection_uptime = Date.now();
              state.socket = socket;
            });
          };
          
          socket.onclose = (event) => {
            console.log('🔌 WebSocket disconnected:', event.code, event.reason);
            set((state) => {
              state.is_connected = false;
              state.session_id = undefined;
              state.socket = null;
              if (event.code !== 1000) {
                state.connection_error = `Connection closed: ${event.reason || 'Unknown error'}`;
              }
            });
            
            // Auto-reconnect if enabled
            if (state.auto_reconnect && event.code !== 1000) {
              setTimeout(() => {
                const currentState = get();
                if (!currentState.is_connected && currentState.reconnect_attempts < 10) {
                  set((state) => {
                    state.reconnect_attempts += 1;
                  });
                  get().connect(url);
                }
              }, 3000);
            }
          };
          
          socket.onerror = (error) => {
            console.error('❌ WebSocket connection error:', error);
            set((state) => {
              state.connection_error = 'Connection error';
              state.reconnect_attempts += 1;
              state.statistics.last_error = 'Connection error';
            });
          };
          
          // Message handler
          socket.onmessage = (event) => {
            try {
              const message = JSON.parse(event.data);
              console.log('📥 Received message:', message);
              
              // Handle connection handshake
              if (message.type === 'connected') {
                console.log('✅ Session established:', message.data);
                set((state) => {
                  state.session_id = message.data.sessionId;
                  state.server_version = message.data.serverVersion;
                });
                return;
              }
              
              // Handle error messages
              if (message.type === 'error') {
                console.error('🚨 Server error:', message.data);
                set((state) => {
                  state.statistics.last_error = message.data.message;
                });
                return;
              }
              
              // Update statistics
              set((state) => {
                state.statistics.messages_received += 1;
                state.statistics.bytes_received += event.data.length;
              });
              
              // Handle pending requests
              if (message.request_id) {
                const currentState = get();
                const request = currentState.pending_requests[message.request_id];
                if (request) {
                  clearTimeout(request.timeout);
                  request.resolve(message.data);
                  set((state) => {
                    delete state.pending_requests[message.request_id];
                  });
                  return;
                }
              }
              
              // Add to message queue for processing
              const serverEvent: ServerEvent = {
                type: message.type,
                data: message.data,
                timestamp: Date.now(),
                session_id: get().session_id,
              };
              
              get().addToMessageQueue(serverEvent);
              
            } catch (error) {
              console.error('❌ Failed to parse WebSocket message:', error);
            }
          };
          
        } catch (error) {
          console.error('❌ Failed to create WebSocket connection:', error);
          set((state) => {
            state.connection_error = error instanceof Error ? error.message : 'Connection failed';
          });
        }
      },
      
      disconnect: () => {
        const state = get();
        
        if (state.socket) {
          state.socket.close(1000, 'Client disconnecting');
          set((state) => {
            state.socket = null;
            state.is_connected = false;
            state.session_id = undefined;
            state.auto_reconnect = false;
          });
        }
      },
      
      send: async <T = any>(event: ClientEvent<T>): Promise<any> => {
        const state = get();
        
        if (!state.socket || !state.is_connected) {
          throw new Error('WebSocket not connected');
        }
        
        // Generate request ID for tracking
        const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        // Create pending request for response tracking
        return new Promise((resolve, reject) => {
          const timeout = setTimeout(() => {
            set((state) => {
              delete state.pending_requests[requestId];
            });
            reject(new Error(`Request timeout: ${event.type}`));
          }, 10000); // 10 second timeout
          
          const pendingRequest: PendingRequest = {
            request_id: requestId,
            type: event.type,
            timestamp: Date.now(),
            timeout,
            resolve,
            reject,
          };
          
          set((state) => {
            state.pending_requests[requestId] = pendingRequest;
          });
          
          // Add request ID to event
          const eventWithId = {
            ...event,
            request_id: requestId,
            session_id: state.session_id,
            timestamp: Date.now(),
          };
          
          // Send message
          try {
            state.socket!.send(JSON.stringify(eventWithId));
            
            // Update statistics
            set((state) => {
              state.statistics.messages_sent += 1;
              state.statistics.bytes_sent += JSON.stringify(eventWithId).length;
            });
            
          } catch (error) {
            set((state) => {
              delete state.pending_requests[requestId];
            });
            clearTimeout(timeout);
            reject(error);
          }
        });
      },
      
      sendMessage: async <T = any>(type: string, data: T): Promise<any> => {
        const event: ClientEvent<T> = {
          type,
          data,
          timestamp: Date.now(),
        };
        return get().send(event);
      },
      
      ping: async (): Promise<boolean> => {
        try {
          await get().sendMessage('client:ping', { timestamp: Date.now() });
          set((state) => {
            state.last_ping = Date.now();
          });
          return true;
        } catch (error) {
          console.error('❌ Ping failed:', error);
          return false;
        }
      },
      
      addToMessageQueue: (event: ServerEvent) => {
        set((state) => {
          state.message_queue.push(event);
        });
      },
      
      updateStatistics: (updates: Partial<ConnectionStatistics>) => {
        set((state) => {
          Object.assign(state.statistics, updates);
        });
      },
      
      clearMessageQueue: () => {
        set((state) => {
          state.message_queue = [];
        });
      },
    })),
    {
      name: 'connection-store',
    }
  )
);

// Helper functions
export const connectionActions = {
  connect: () => useConnectionStore.getState().connect(),
  disconnect: () => useConnectionStore.getState().disconnect(),
  ping: () => useConnectionStore.getState().ping(),
  sendMessage: (type: string, data: any) => useConnectionStore.getState().sendMessage(type, data),
};