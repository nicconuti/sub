import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import SplViewer from '../src/components/visualization/SplViewer';

vi.mock('react-plotly.js', () => ({ default: () => <div data-testid="plot" /> }));
vi.mock('../src/store/simulation', () => ({ useSimulationStore: () => ({ currentProject: null }) }));

describe('SplViewer', () => {
  it('renders without data', () => {
    render(<SplViewer sources={[]} viewState={{ zoom:1, pan:[0,0], center:[0,0], rotation:0 }} onViewStateChange={() => {}} />);
    expect(screen.getByTestId('plot')).toBeInTheDocument();
  });
});
