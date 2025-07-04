import { describe, it, expect } from 'vitest';
import { getSplColor, getFrequencyColor } from '../src/theme/audioTheme';

describe('audio theme utilities', () => {
  it('getSplColor returns low color', () => {
    expect(getSplColor(60)).toMatch('#2E7D32');
  });

  it('getFrequencyColor returns bass color', () => {
    expect(getFrequencyColor(100)).toMatch('#3F51B5');
  });
});
