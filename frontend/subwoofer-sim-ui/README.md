# 🔊 Subwoofer Simulation UI

Professional React frontend for acoustic engineering and subwoofer array optimization.

## ✨ Features

- **Professional Audio Design**: Dark theme optimized for acoustic engineering
- **Real-time WebSocket Communication**: Live SPL updates and optimization progress
- **Interactive 3D Visualization**: Plotly.js-powered SPL mapping
- **Advanced Source Management**: Drag & drop subwoofer positioning
- **Genetic Algorithm Optimization**: AI-powered array optimization
- **Responsive Layout**: Professional sidebar and canvas interface

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Type checking
npm run type-check
```

## 🏗️ Architecture

### Core Technologies
- **React 18** with TypeScript
- **Material-UI v5** for professional components
- **Zustand** for state management
- **Plotly.js** for scientific visualization
- **Socket.IO** for WebSocket communication
- **Framer Motion** for smooth animations

### Project Structure
```
src/
├── components/
│   ├── layout/           # AppLayout, Header, Sidebar
│   ├── panels/           # Control panels (Simulation, Sources, etc.)
│   ├── visualization/    # SPL viewer, plotting components
│   └── common/           # Shared UI components
├── store/
│   ├── simulation.ts     # Simulation state management
│   └── connection.ts     # WebSocket connection management
├── theme/
│   └── audioTheme.ts     # Professional audio theme
├── types/
│   └── index.ts          # TypeScript definitions
└── App.tsx               # Main application
```

## 🎨 Design System

### Color Palette
- **Primary**: `#00D4FF` (Cyan) - Audio spectrum visualization
- **Secondary**: `#FF6B35` (Orange) - Warnings and controls
- **Background**: `#0A0A0A` - Professional studio black
- **Surfaces**: Dark grays with subtle transparency

### Typography
- **Primary**: Inter font family for exceptional readability
- **Monospace**: SF Mono for technical values and data
- **Optimized sizes**: Designed for technical content

### Audio-Specific Colors
- **SPL Levels**: Green (low) → Orange (moderate) → Red (high) → Purple (critical)
- **Frequency Bands**: Purple (sub) → Blue (bass) → Green (mid) → Orange (high) → Red (ultra)
- **Phase**: Green (in-phase) → Red (out-of-phase) → Amber (neutral)

## 📡 WebSocket Integration

### Real-time Events
```typescript
// Client → Server
'simulation:start'          // Start SPL calculation
'source:move'              // Drag & drop source positioning
'optimization:start'       // Begin genetic algorithm

// Server → Client  
'simulation:progress'      // Live progress updates
'simulation:spl_chunk'     // Progressive SPL map loading
'optimization:generation'  // GA generation updates
```

### Performance Optimizations
- **Chunked Data Transfer**: Large SPL maps sent in progressive chunks
- **Binary Compression**: msgpack for large datasets
- **Debounced Updates**: Prevents overwhelming with rapid parameter changes
- **Connection Health**: Auto-reconnect with exponential backoff

## 🔧 Development

### Environment Setup
```bash
# Required Node.js version
node --version  # >= 18.0.0

# Install dependencies
npm install

# Start with hot reload
npm run dev
```

### Code Quality
```bash
# Linting
npm run lint

# Type checking
npm run type-check

# Build verification
npm run build
```

### WebSocket Backend
Ensure the FastAPI backend is running:
```bash
# In backend directory
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📊 Performance

### Metrics
- **First Load**: ~2-3 seconds
- **WebSocket Latency**: ~50ms
- **SPL Calculation**: Real-time streaming
- **Memory Usage**: Optimized for large datasets

### Optimizations
- **Code Splitting**: Automatic route-based splitting
- **Tree Shaking**: Dead code elimination
- **Compression**: Gzip/Brotli for production
- **Caching**: Efficient state management

## 🎯 Professional Features

### SPL Visualization
- Multiple professional colorscales (Spectrum, Thermal, Acoustic)
- Real-time hover previews
- Interactive zoom and pan
- Export capabilities (PNG, SVG, PDF)

### Source Management
- Drag & drop positioning
- Real-time parameter validation
- Bulk operations
- Visual feedback with SPL-based coloring

### Optimization
- Live genetic algorithm visualization
- Convergence monitoring
- Real-time fitness updates
- Result comparison tools

## 🤝 Contributing

1. Follow the established code style
2. Maintain professional visual design
3. Ensure WebSocket compatibility
4. Test on multiple screen sizes
5. Update TypeScript definitions

---

**Built with ❤️ for acoustic engineers and audio professionals**
```
