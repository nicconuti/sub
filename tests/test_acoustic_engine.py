#!/usr/bin/env python3
"""
Test acoustic engine to isolate the Numba compilation issue
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.acoustic_engine import calculate_spl_vectorized
from core.config import SUB_DTYPE

def test_minimal_acoustic_calculation():
    """Test minimal acoustic calculation to identify Numba issue"""
    
    # Create a simple test case
    px = np.array([1.0, 2.0, 3.0])  # Test points X
    py = np.array([1.0, 2.0, 3.0])  # Test points Y
    
    # Create a simple sources array with correct SUB_DTYPE structure
    p_ref = 20e-6
    test_spl = 105.0  # dB SPL
    pressure_val = p_ref * (10**(test_spl / 20.0))
    
    sources_array = np.zeros(2, dtype=SUB_DTYPE)
    sources_array[0] = (0.0, 0.0, pressure_val, 1.0, 0.0, 0.0, 1)  # source 1
    sources_array[1] = (5.0, 5.0, pressure_val, 1.0, 0.0, 0.0, 1)  # source 2
    
    print(f"Sources array shape: {sources_array.shape}")
    print(f"Sources array dtype: {sources_array.dtype}")
    print(f"Sources array:\n{sources_array}")
    
    # Test parameters
    frequency = 80.0  # Hz
    speed_of_sound = 343.0  # m/s
    
    try:
        print("\n🧪 Testing acoustic calculation...")
        result = calculate_spl_vectorized(px, py, frequency, speed_of_sound, sources_array)
        print(f"✅ Success! Result: {result}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_len_on_structured_array():
    """Test if len() works on structured arrays in Numba context"""
    
    sources_array = np.zeros(3, dtype=SUB_DTYPE)
    
    print(f"\n🧪 Testing len() on structured array...")
    print(f"Python len(): {len(sources_array)}")
    print(f"NumPy shape[0]: {sources_array.shape[0]}")
    
    # Test if len() works in numba context (this might be the issue)
    from numba import jit
    
    @jit(nopython=True)
    def test_len_in_numba(arr):
        return len(arr)
    
    try:
        result = test_len_in_numba(sources_array)
        print(f"✅ len() in Numba works: {result}")
        return True
    except Exception as e:
        print(f"❌ len() in Numba failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Acoustic Engine Issues")
    print("=" * 50)
    
    # Test 1: len() on structured array
    test_len_on_structured_array()
    
    # Test 2: Full acoustic calculation
    test_minimal_acoustic_calculation()
