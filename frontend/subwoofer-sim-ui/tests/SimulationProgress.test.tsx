import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import SimulationProgress from '../src/components/common/SimulationProgress';

const baseProgress = {
  progress: 50,
  current_step: 'calculating_spl',
  estimated_time: 120,
  chunks_received: 1,
  total_chunks: 2,
};

describe('SimulationProgress', () => {
  it('renders full variant by default', () => {
    render(<SimulationProgress progress={baseProgress} />);
    expect(screen.getByText(/calculating/i)).toBeInTheDocument();
  });

  it('renders compact variant', () => {
    render(<SimulationProgress progress={baseProgress} variant="compact" />);
    expect(screen.getByText(/50/)).toBeInTheDocument();
  });
});
