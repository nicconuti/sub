import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ResultsPanel } from '../src/components/panels/ResultsPanel';

describe('ResultsPanel', () => {
  it('shows coming soon message', () => {
    render(<ResultsPanel />);
    expect(screen.getByText(/Coming Soon/i)).toBeInTheDocument();
  });
});
