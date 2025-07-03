"""Data loading and file operations for subwoofer simulation."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and saving of project data."""
    
    def __init__(self):
        self.supported_formats = ['.xlsx', '.json', '.csv']
    
    def load_excel_project(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load project data from Excel file.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Dictionary containing project data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix not in ['.xlsx', '.xls']:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        try:
            # Load all sheets
            excel_data = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
            
            project_data = {
                'metadata': {
                    'file_path': str(file_path),
                    'sheets': list(excel_data.keys())
                },
                'data': {}
            }
            
            # Process each sheet
            for sheet_name, df in excel_data.items():
                project_data['data'][sheet_name] = df.to_dict('records')
            
            logger.info(f"Loaded project from {file_path}")
            return project_data
            
        except Exception as e:
            logger.error(f"Error loading Excel file {file_path}: {e}")
            raise
    
    def save_excel_project(self, project_data: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """Save project data to Excel file.
        
        Args:
            project_data: Dictionary containing project data
            file_path: Path to save Excel file
        """
        file_path = Path(file_path)
        
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                for sheet_name, data in project_data.get('data', {}).items():
                    if isinstance(data, list) and data:
                        df = pd.DataFrame(data)
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            logger.info(f"Saved project to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving Excel file {file_path}: {e}")
            raise
    
    def load_json_config(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary containing configuration data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            logger.info(f"Loaded config from {file_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading JSON config {file_path}: {e}")
            raise
    
    def save_json_config(self, config: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file.
        
        Args:
            config: Dictionary containing configuration data
            file_path: Path to save JSON file
        """
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved config to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving JSON config {file_path}: {e}")
            raise
    
    def load_csv_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame containing CSV data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info(f"Loaded CSV data from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            raise
    
    def get_file_list(self, directory: Union[str, Path], pattern: str = "*") -> List[Path]:
        """Get list of files matching pattern in directory.
        
        Args:
            directory: Directory to search
            pattern: File pattern (default: "*")
            
        Returns:
            List of matching file paths
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        files = list(directory.glob(pattern))
        logger.info(f"Found {len(files)} files matching '{pattern}' in {directory}")
        
        return files
    
    def validate_file_format(self, file_path: Union[str, Path]) -> bool:
        """Validate if file format is supported.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if format is supported, False otherwise
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_formats