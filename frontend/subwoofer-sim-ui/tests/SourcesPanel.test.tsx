import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import SourcesPanel from '../src/components/panels/SourcesPanel';

vi.mock('../src/store/simulation', () => ({
  useCurrentSources: () => [],
  useSelectedSources: () => [],
  useSimulationStore: () => ({
    addSource: vi.fn(),
    updateSource: vi.fn(),
    removeSource: vi.fn(),
    selectSources: vi.fn(),
  }),
}));

describe('SourcesPanel', () => {
  it('renders no sources message', () => {
    render(<SourcesPanel />);
    expect(screen.getByText(/No sources added/i)).toBeInTheDocument();
  });
});
