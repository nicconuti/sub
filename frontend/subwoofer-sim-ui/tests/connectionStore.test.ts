import { describe, it, expect } from 'vitest';
import { useConnectionStore } from '../src/store/connection';

describe('connection store', () => {
  it('has initial disconnected state', () => {
    const state = useConnectionStore.getState();
    expect(state.is_connected).toBe(false);
  });
});
