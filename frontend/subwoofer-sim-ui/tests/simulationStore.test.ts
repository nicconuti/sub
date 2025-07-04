import { describe, it, expect } from 'vitest';
import { useSimulationStore } from '../src/store/simulation';

describe('simulation store', () => {
  it('creates default project', () => {
    useSimulationStore.getState().createDefaultProject();
    const state = useSimulationStore.getState();
    expect(state.currentProject).not.toBeNull();
  });
});
