import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import AppLayout from '../src/components/layout/AppLayout';

vi.mock('../src/store/simulation', () => ({
  useSimulationStore: () => ({
    currentProject: null,
    isSimulating: false,
    simulationProgress: { progress: 0, current_step: 'idle' },
    startSimulation: vi.fn(),
    stopSimulation: vi.fn(),
  }),
}));

vi.mock('../src/store/connection', () => ({
  useConnectionStore: () => ({ isConnected: true, connectionError: null, serverVersion: '1.0' }),
}));

describe('AppLayout', () => {
  it('renders children', () => {
    render(
      <AppLayout>
        <div data-testid="child" />
      </AppLayout>
    );
    expect(screen.getByTestId('child')).toBeInTheDocument();
  });
});
