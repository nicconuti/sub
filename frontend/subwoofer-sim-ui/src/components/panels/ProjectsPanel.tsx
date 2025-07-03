/**
 * Projects Panel
 * Project management and file operations
 */

import React from 'react';
import { Box, Typography, Card, CardContent, Button } from '@mui/material';
import { FolderOpen as ProjectsIcon } from '@mui/icons-material';

export const ProjectsPanel: React.FC = () => {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <ProjectsIcon sx={{ color: 'primary.main' }} />
            <Typography variant="h6">Projects</Typography>
          </Box>
          
          <Typography variant="body2" color="text.secondary" paragraph>
            Project management, save/load, and import/export functionality.
          </Typography>
          
          <Button variant="contained" disabled>
            Coming Soon
          </Button>
        </CardContent>
      </Card>
    </Box>
  );
};