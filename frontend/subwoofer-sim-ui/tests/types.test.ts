import { describe, it, expect } from 'vitest';
import { isSourceData, isServerEvent, isSplMapData } from '../src/types';

describe('type guards', () => {
  it('validates SourceData', () => {
    expect(isSourceData({ id:'1', x:0, y:0, spl_rms:80, gain_db:0, delay_ms:0, angle:0, polarity:1 })).toBe(true);
  });
  it('invalid SourceData returns false', () => {
    expect(isSourceData({})).toBe(false);
  });
  it('validates ServerEvent', () => {
    expect(isServerEvent({ type:'connected', timestamp:0, data:{} })).toBe(true);
  });
  it('validates SplMapData', () => {
    expect(isSplMapData({ X:[[0]], Y:[[0]], SPL:[[0]], frequency:80, timestamp:0 })).toBe(true);
  });
});
