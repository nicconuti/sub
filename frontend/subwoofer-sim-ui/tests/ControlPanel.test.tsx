import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import ControlPanel from '../src/components/panels/ControlPanel';

vi.mock('../src/components/panels/SimulationPanel', () => ({ SimulationPanel: () => <div>SimulationPanel</div> }));
vi.mock('../src/components/panels/SourcesPanel', () => ({ SourcesPanel: () => <div>SourcesPanel</div> }));
vi.mock('../src/components/panels/OptimizationPanel', () => ({ OptimizationPanel: () => <div>OptimizationPanel</div> }));
vi.mock('../src/components/panels/ProjectsPanel', () => ({ ProjectsPanel: () => <div>ProjectsPanel</div> }));
vi.mock('../src/components/panels/ResultsPanel', () => ({ ResultsPanel: () => <div>ResultsPanel</div> }));

vi.mock('../src/store/simulation', () => ({ useSimulationStore: () => ({}) }));


describe('ControlPanel', () => {
  it('renders tab labels', () => {
    render(<ControlPanel />);
    expect(screen.getByRole('tab', { name: /Simulation/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Sources/i })).toBeInTheDocument();
  });
});
