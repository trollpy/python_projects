"""
Data loading utilities with chunked processing support
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Iterator, List
import os

logger = logging.getLogger(__name__)


class DataLoader:
    """Handle data loading with support for large files"""
    
    def __init__(self, filepath: str, chunk_size: int = 10000):
        self.filepath = filepath
        self.chunk_size = chunk_size
        self._validate_file()
    
    def _validate_file(self) -> None:
        """Validate that the data file exists"""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Data file not found: {self.filepath}\n"
                f"Please download the dataset from: "
                f"https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge"
            )
    
    def load_full(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load entire dataset or sample into memory
        
        Args:
            sample_size: Number of rows to load (None for all)
        
        Returns:
            DataFrame with loaded data
        """
        try:
            logger.info(f"Loading data from {self.filepath}")
            
            if sample_size:
                df = pd.read_csv(self.filepath, nrows=sample_size, low_memory=False)
                logger.info(f"Loaded {len(df)} rows (sample)")
            else:
                df = pd.read_csv(self.filepath, low_memory=False)
                logger.info(f"Loaded {len(df)} rows (full dataset)")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def load_chunks(self) -> Iterator[pd.DataFrame]:
        """
        Load data in chunks for memory-efficient processing
        
        Yields:
            DataFrame chunks
        """
        try:
            logger.info(f"Loading data in chunks of {self.chunk_size}")
            
            for chunk in pd.read_csv(
                self.filepath, 
                chunksize=self.chunk_size,
                low_memory=False
            ):
                yield chunk
        
        except Exception as e:
            logger.error(f"Error loading data chunks: {e}")
            raise
    
    def get_column_names(self) -> List[str]:
        """Get column names without loading full dataset"""
        try:
            df_sample = pd.read_csv(self.filepath, nrows=1)
            return df_sample.columns.tolist()
        except Exception as e:
            logger.error(f"Error reading column names: {e}")
            raise
    
    def get_file_info(self) -> dict:
        """Get information about the data file"""
        try:
            file_size = os.path.getsize(self.filepath)
            file_size_mb = file_size / (1024 * 1024)
            
            # Get row count efficiently
            with open(self.filepath, 'r', encoding='utf-8') as f:
                row_count = sum(1 for _ in f) - 1  # Subtract header
            
            return {
                'filepath': self.filepath,
                'size_mb': round(file_size_mb, 2),
                'estimated_rows': row_count,
                'exists': True
            }
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {
                'filepath': self.filepath,
                'exists': False,
                'error': str(e)
            }