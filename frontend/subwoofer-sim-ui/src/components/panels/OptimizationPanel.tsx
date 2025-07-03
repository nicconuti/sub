/**
 * Optimization Panel
 * Genetic algorithm optimization controls
 */

import React from 'react';
import { Box, Typography, Card, CardContent, Button } from '@mui/material';
import { TuneRounded as OptimizationIcon } from '@mui/icons-material';

export const OptimizationPanel: React.FC = () => {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <OptimizationIcon sx={{ color: 'primary.main' }} />
            <Typography variant="h6">Optimization</Typography>
          </Box>
          
          <Typography variant="body2" color="text.secondary" paragraph>
            Genetic algorithm optimization for source placement and configuration.
          </Typography>
          
          <Button variant="contained" disabled>
            Coming Soon
          </Button>
        </CardContent>
      </Card>
    </Box>
  );
};