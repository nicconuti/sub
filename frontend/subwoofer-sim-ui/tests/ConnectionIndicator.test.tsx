import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import ConnectionIndicator from '../src/components/common/ConnectionIndicator';


describe('ConnectionIndicator', () => {
  it('renders connected state', () => {
    render(<ConnectionIndicator isConnected={true} />);
    expect(screen.getByText(/Connected/i)).toBeInTheDocument();
  });

  it('renders disconnected state', () => {
    render(<ConnectionIndicator isConnected={false} reconnectAttempts={0} />);
    expect(screen.getByText(/Offline/i)).toBeInTheDocument();
  });
});
