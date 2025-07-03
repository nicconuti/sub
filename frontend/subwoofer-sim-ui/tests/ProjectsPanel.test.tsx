import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ProjectsPanel } from '../src/components/panels/ProjectsPanel';

describe('ProjectsPanel', () => {
  it('shows coming soon message', () => {
    render(<ProjectsPanel />);
    expect(screen.getByText(/Coming Soon/i)).toBeInTheDocument();
  });
});
