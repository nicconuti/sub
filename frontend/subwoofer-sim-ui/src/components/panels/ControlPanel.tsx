/**
 * Professional Control Panel
 * Main control interface for simulation parameters and source management
 */

import React, { useState } from 'react';
import {
  Box,
  Tabs,
  Tab,
  Divider,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Settings as SimulationIcon,
  GraphicEq as SourcesIcon,
  TuneRounded as OptimizationIcon,
  FolderOpen as ProjectsIcon,
  Assessment as ResultsIcon,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';

import { SimulationPanel } from './SimulationPanel';
import { SourcesPanel } from './SourcesPanel';
import { OptimizationPanel } from './OptimizationPanel';
import { ProjectsPanel } from './ProjectsPanel';
import { ResultsPanel } from './ResultsPanel';
// Temporary constants
const UI_CONSTANTS = {
  SIDEBAR_WIDTH: 320,
  HEADER_HEIGHT: 64,
  TAB_HEIGHT: 48,
};

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div
    role="tabpanel"
    hidden={value !== index}
    id={`control-tabpanel-${index}`}
    aria-labelledby={`control-tab-${index}`}
    style={{ height: '100%' }}
  >
    {value === index && (
      <AnimatePresence mode="wait">
        <motion.div
          key={index}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2 }}
          style={{ height: '100%' }}
        >
          {children}
        </motion.div>
      </AnimatePresence>
    )}
  </div>
);

const a11yProps = (index: number) => ({
  id: `control-tab-${index}`,
  'aria-controls': `control-tabpanel-${index}`,
});

export const ControlPanel: React.FC = () => {
  const theme = useTheme();
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const tabConfig = [
    {
      label: 'Simulation',
      icon: SimulationIcon,
      component: SimulationPanel,
    },
    {
      label: 'Sources',
      icon: SourcesIcon,
      component: SourcesPanel,
    },
    {
      label: 'Optimization',
      icon: OptimizationIcon,
      component: OptimizationPanel,
    },
    {
      label: 'Projects',
      icon: ProjectsIcon,
      component: ProjectsPanel,
    },
    {
      label: 'Results',
      icon: ResultsIcon,
      component: ResultsPanel,
    },
  ];

  return (
    <Box
      sx={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'background.paper',
      }}
    >
      {/* Professional Tab Navigation */}
      <Box
        sx={{
          borderBottom: `1px solid ${theme.palette.divider}`,
          bgcolor: alpha(theme.palette.background.paper, 0.8),
          backdropFilter: 'blur(10px)',
        }}
      >
        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          orientation="horizontal"
          variant="scrollable"
          scrollButtons="auto"
          sx={{
            minHeight: UI_CONSTANTS.TAB_HEIGHT,
            '& .MuiTabs-indicator': {
              height: 3,
              borderRadius: '3px 3px 0 0',
              background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
            },
            '& .MuiTabs-scrollButtons': {
              color: theme.palette.text.secondary,
              '&.Mui-disabled': {
                opacity: 0.3,
              },
            },
          }}
        >
          {tabConfig.map((tab, index) => {
            const IconComponent = tab.icon;
            return (
              <Tab
                key={tab.label}
                icon={
                  <motion.div
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <IconComponent />
                  </motion.div>
                }
                label={tab.label}
                iconPosition="start"
                {...a11yProps(index)}
                sx={{
                  minHeight: UI_CONSTANTS.TAB_HEIGHT,
                  textTransform: 'none',
                  fontWeight: 500,
                  fontSize: '0.875rem',
                  gap: 1,
                  px: 2,
                  color: activeTab === index ? 'primary.main' : 'text.secondary',
                  transition: theme.transitions.create(['color', 'background-color'], {
                    duration: theme.transitions.duration.short,
                  }),
                  '&:hover': {
                    bgcolor: alpha(theme.palette.primary.main, 0.04),
                    color: 'primary.main',
                  },
                  '&.Mui-selected': {
                    color: 'primary.main',
                    bgcolor: alpha(theme.palette.primary.main, 0.08),
                  },
                  '& .MuiTab-iconWrapper': {
                    fontSize: '1.25rem',
                  },
                }}
              />
            );
          })}
        </Tabs>
      </Box>

      {/* Tab Content Area */}
      <Box
        sx={{
          flexGrow: 1,
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        {tabConfig.map((tab, index) => {
          const ComponentToRender = tab.component;
          return (
            <TabPanel key={tab.label} value={activeTab} index={index}>
              <Box
                sx={{
                  height: '100%',
                  overflow: 'auto',
                  p: 2,
                  
                  // Professional scrollbar styling
                  '&::-webkit-scrollbar': {
                    width: 6,
                  },
                  '&::-webkit-scrollbar-track': {
                    bgcolor: 'transparent',
                  },
                  '&::-webkit-scrollbar-thumb': {
                    bgcolor: alpha(theme.palette.text.secondary, 0.3),
                    borderRadius: 3,
                    '&:hover': {
                      bgcolor: alpha(theme.palette.text.secondary, 0.5),
                    },
                  },
                }}
              >
                <ComponentToRender />
              </Box>
            </TabPanel>
          );
        })}
      </Box>
    </Box>
  );
};

export default ControlPanel;