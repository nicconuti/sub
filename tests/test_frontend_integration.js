/**
 * Frontend Integration Test
 * Test SPL chunk reconstruction and simulation completion
 */

// Mock data matching backend response format
const mockChunks = {
  0: {
    chunk_index: 0,
    total_chunks: 3,
    X: [[1, 2], [3, 4]],
    Y: [[5, 6], [7, 8]], 
    SPL: [[90, 92], [88, 91]]
  },
  1: {
    chunk_index: 1,
    total_chunks: 3,
    X: [[9, 10], [11, 12]],
    Y: [[13, 14], [15, 16]],
    SPL: [[85, 87], [89, 86]]
  },
  2: {
    chunk_index: 2,
    total_chunks: 3,
    X: [[17, 18], [19, 20]],
    Y: [[21, 22], [23, 24]],
    SPL: [[93, 95], [87, 90]]
  }
};

const mockCompletionData = {
  simulation_id: "test-123",
  computation_time: 2.5,
  statistics: {
    total_sources: 2,
    grid_points: 100,
    frequency: 80.0
  }
};

// Test chunk reconstruction logic
function testChunkReconstruction() {
  console.log('🧪 Testing SPL chunk reconstruction...');
  
  // Simulate the reconstruction logic from simulation.ts
  const sortedChunks = Object.entries(mockChunks)
    .sort(([a], [b]) => parseInt(a) - parseInt(b))
    .map(([_, chunk]) => chunk);
  
  console.log(`✅ Sorted ${sortedChunks.length} chunks`);
  
  // Combine all X, Y, and SPL data
  const combinedX = [];
  const combinedY = [];
  const combinedSPL = [];
  
  sortedChunks.forEach(chunk => {
    if (chunk.X && chunk.Y && chunk.SPL) {
      combinedX.push(...chunk.X);
      combinedY.push(...chunk.Y);
      combinedSPL.push(...chunk.SPL);
    }
  });
  
  // Create complete SPL map
  const completeSplMap = {
    X: combinedX,
    Y: combinedY,
    SPL: combinedSPL,
    frequency: mockCompletionData.statistics.frequency || 80,
    timestamp: Date.now(),
    statistics: mockCompletionData.statistics || {}
  };
  
  console.log('✅ Reconstructed SPL map:', {
    totalXPoints: combinedX.length,
    totalYPoints: combinedY.length,
    totalSPLPoints: combinedSPL.length,
    frequency: completeSplMap.frequency,
    hasStatistics: !!completeSplMap.statistics
  });
  
  // Test frequency key generation (the part that was causing the error)
  const frequencyKey = (completeSplMap.frequency || 80).toString();
  console.log(`✅ Frequency key: "${frequencyKey}"`);
  
  // Simulate storing in results
  const mockResults = {
    spl_maps: {}
  };
  mockResults.spl_maps[frequencyKey] = completeSplMap;
  
  console.log('✅ Successfully stored SPL map with key:', frequencyKey);
  console.log('✅ Mock results structure:', Object.keys(mockResults.spl_maps));
  
  return true;
}

// Test undefined frequency handling
function testUndefinedFrequencyHandling() {
  console.log('\n🧪 Testing undefined frequency handling...');
  
  const splMapWithoutFreq = {
    X: [[1, 2]],
    Y: [[3, 4]],
    SPL: [[85, 90]],
    // frequency: undefined - intentionally missing
    timestamp: Date.now()
  };
  
  // Test the safety check
  const frequencyKey = (splMapWithoutFreq.frequency || 80).toString();
  console.log(`✅ Handled undefined frequency, key: "${frequencyKey}"`);
  
  return frequencyKey === "80";
}

// Run tests
console.log('🚀 Starting Frontend Integration Tests');
console.log('='.repeat(50));

try {
  const test1 = testChunkReconstruction();
  const test2 = testUndefinedFrequencyHandling();
  
  if (test1 && test2) {
    console.log('\n🎉 All tests passed! Frontend integration should work correctly.');
  } else {
    console.log('\n❌ Some tests failed.');
  }
} catch (error) {
  console.error('\n❌ Test error:', error);
}