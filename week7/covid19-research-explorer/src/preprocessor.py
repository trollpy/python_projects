"""
Data preprocessing and cleaning utilities
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, List
import re

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data cleaning and preprocessing"""
    
    def __init__(self):
        self.cleaning_stats = {}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main cleaning pipeline
        
        Args:
            df: Raw DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning pipeline")
        
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        # Clean dates
        df_clean = self._clean_dates(df_clean)
        
        # Clean text fields
        df_clean = self._clean_text_fields(df_clean)
        
        # Remove invalid rows
        df_clean = self._remove_invalid_rows(df_clean)
        
        # Add derived features
        df_clean = self._add_derived_features(df_clean)
        
        # Record cleaning statistics
        self.cleaning_stats = {
            'initial_rows': initial_rows,
            'final_rows': len(df_clean),
            'removed_rows': initial_rows - len(df_clean),
            'removal_percentage': ((initial_rows - len(df_clean)) / initial_rows) * 100
        }
        
        logger.info(f"Cleaning complete. Removed {self.cleaning_stats['removed_rows']} rows "
                   f"({self.cleaning_stats['removal_percentage']:.2f}%)")
        
        return df_clean
    
    def _clean_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert and clean date fields"""
        if 'publish_time' in df.columns:
            df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
            df['year'] = df['publish_time'].dt.year
            df['month'] = df['publish_time'].dt.month
            df['date'] = df['publish_time'].dt.date
            
            # Filter out invalid years
            current_year = pd.Timestamp.now().year
            df = df[(df['year'] >= 1900) & (df['year'] <= current_year)]
        
        return df
    
    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text fields"""
        text_columns = ['title', 'abstract', 'authors', 'journal']
        
        for col in text_columns:
            if col in df.columns:
                # Strip whitespace
                df[col] = df[col].astype(str).str.strip()
                
                # Replace 'nan' string with actual NaN
                df[col] = df[col].replace(['nan', 'NaN', 'None', ''], np.nan)
        
        return df
    
    def _remove_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with critical missing data"""
        # Must have title
        if 'title' in df.columns:
            df = df.dropna(subset=['title'])
        
        # Remove duplicates based on title
        if 'title' in df.columns:
            df = df.drop_duplicates(subset=['title'], keep='first')
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed features"""
        # Word counts
        if 'abstract' in df.columns:
            df['abstract_word_count'] = df['abstract'].fillna('').astype(str).apply(
                lambda x: len(x.split())
            )
            df['has_abstract'] = df['abstract'].notna()
        
        if 'title' in df.columns:
            df['title_word_count'] = df['title'].fillna('').astype(str).apply(
                lambda x: len(x.split())
            )
        
        # Author count
        if 'authors' in df.columns:
            df['author_count'] = df['authors'].fillna('').astype(str).apply(
                lambda x: len([a for a in x.split(';') if a.strip()])
            )
        
        # Has DOI
        if 'doi' in df.columns:
            df['has_doi'] = df['doi'].notna()
        
        return df
    
    def filter_by_date_range(
        self, 
        df: pd.DataFrame, 
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> pd.DataFrame:
        """Filter DataFrame by year range"""
        if 'year' not in df.columns:
            logger.warning("No 'year' column found for filtering")
            return df
        
        df_filtered = df.copy()
        
        if start_year:
            df_filtered = df_filtered[df_filtered['year'] >= start_year]
        
        if end_year:
            df_filtered = df_filtered[df_filtered['year'] <= end_year]
        
        return df_filtered
    
    def filter_by_sources(
        self, 
        df: pd.DataFrame, 
        sources: List[str]
    ) -> pd.DataFrame:
        """Filter DataFrame by source list"""
        if 'source_x' not in df.columns:
            logger.warning("No 'source_x' column found for filtering")
            return df
        
        return df[df['source_x'].isin(sources)]
    
    def search_text(
        self, 
        df: pd.DataFrame, 
        query: str,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Search for text in specified columns"""
        if not query:
            return df
        
        if columns is None:
            columns = ['title', 'abstract']
        
        # Build search mask
        mask = pd.Series([False] * len(df))
        
        for col in columns:
            if col in df.columns:
                mask |= df[col].astype(str).str.contains(
                    query, 
                    case=False, 
                    na=False,
                    regex=False
                )
        
        return df[mask]