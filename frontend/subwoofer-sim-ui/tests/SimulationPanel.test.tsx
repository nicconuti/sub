import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import SimulationPanel from '../src/components/panels/SimulationPanel';

vi.mock('../src/store/simulation', () => ({
  useSimulationStore: () => ({
    simulationParams: { frequency: 80, speed_of_sound: 343, grid_resolution: 0.1, room_vertices: [] },
    updateSimulationParams: vi.fn(),
    isSimulating: false,
    startSimulation: vi.fn(),
    stopSimulation: vi.fn(),
  }),
}));

describe('SimulationPanel', () => {
  it('renders frequency settings', () => {
    render(<SimulationPanel />);
    expect(screen.getByText(/Frequency Settings/i)).toBeInTheDocument();
  });
});
