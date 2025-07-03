import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { OptimizationPanel } from '../src/components/panels/OptimizationPanel';

describe('OptimizationPanel', () => {
  it('shows coming soon message', () => {
    render(<OptimizationPanel />);
    expect(screen.getByText(/Coming Soon/i)).toBeInTheDocument();
  });
});
