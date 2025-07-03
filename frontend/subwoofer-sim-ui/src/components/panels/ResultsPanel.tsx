/**
 * Results Panel
 * Analysis and export of simulation results
 */

import React from 'react';
import { Box, Typography, Card, CardContent, Button } from '@mui/material';
import { Assessment as ResultsIcon } from '@mui/icons-material';

export const ResultsPanel: React.FC = () => {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <ResultsIcon sx={{ color: 'primary.main' }} />
            <Typography variant="h6">Results</Typography>
          </Box>
          
          <Typography variant="body2" color="text.secondary" paragraph>
            Simulation results analysis, statistics, and export options.
          </Typography>
          
          <Button variant="contained" disabled>
            Coming Soon
          </Button>
        </CardContent>
      </Card>
    </Box>
  );
};