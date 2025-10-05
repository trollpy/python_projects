"""
Analysis functions for COVID-19 research data
"""
import pandas as pd
import numpy as np
from collections import Counter
import re
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class COVID19Analyzer:
    """Perform various analyses on COVID-19 research data"""
    
    def __init__(self, stop_words: Optional[List[str]] = None):
        self.stop_words = set(stop_words) if stop_words else set()
    
    def get_publication_trends(self, df: pd.DataFrame) -> pd.Series:
        """Get publication counts by year"""
        if 'year' not in df.columns:
            logger.warning("No 'year' column found")
            return pd.Series()
        
        return df['year'].value_counts().sort_index()
    
    def get_monthly_trends(self, df: pd.DataFrame) -> pd.Series:
        """Get publication counts by month"""
        if 'publish_time' not in df.columns:
            logger.warning("No 'publish_time' column found")
            return pd.Series()
        
        df['year_month'] = df['publish_time'].dt.to_period('M')
        return df['year_month'].value_counts().sort_index()
    
    def get_top_journals(self, df: pd.DataFrame, top_n: int = 15) -> pd.Series:
        """Get top N journals by publication count"""
        if 'journal' not in df.columns:
            logger.warning("No 'journal' column found")
            return pd.Series()
        
        return df['journal'].value_counts().head(top_n)
    
    def get_top_sources(self, df: pd.DataFrame, top_n: int = 10) -> pd.Series:
        """Get top N sources by publication count"""
        if 'source_x' not in df.columns:
            logger.warning("No 'source_x' column found")
            return pd.Series()
        
        return df['source_x'].value_counts().head(top_n)
    
    def get_common_words(
        self, 
        text_series: pd.Series, 
        top_n: int = 30,
        min_length: int = 4
    ) -> List[Tuple[str, int]]:
        """
        Extract most common words from text
        
        Args:
            text_series: Series containing text
            top_n: Number of top words to return
            min_length: Minimum word length
        
        Returns:
            List of (word, count) tuples
        """
        # Combine all text
        all_text = ' '.join(text_series.dropna().astype(str).tolist())
        
        # Clean and tokenize
        words = re.findall(rf'\b[a-z]{{{min_length},}}\b', all_text.lower())
        
        # Remove stop words
        words = [w for w in words if w not in self.stop_words]
        
        # Count words
        word_counts = Counter(words)
        return word_counts.most_common(top_n)
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Get comprehensive summary statistics"""
        stats = {
            'total_papers': len(df),
            'unique_journals': df['journal'].nunique() if 'journal' in df.columns else 0,
            'unique_sources': df['source_x'].nunique() if 'source_x' in df.columns else 0,
            'date_range': {
                'min_year': int(df['year'].min()) if 'year' in df.columns else None,
                'max_year': int(df['year'].max()) if 'year' in df.columns else None
            }
        }
        
        # Abstract statistics
        if 'abstract' in df.columns:
            stats['papers_with_abstract'] = df['abstract'].notna().sum()
            stats['abstract_coverage_pct'] = (stats['papers_with_abstract'] / len(df)) * 100
        
        if 'abstract_word_count' in df.columns:
            stats['avg_abstract_length'] = df['abstract_word_count'].mean()
            stats['median_abstract_length'] = df['abstract_word_count'].median()
        
        # Author statistics
        if 'author_count' in df.columns:
            stats['avg_authors_per_paper'] = df['author_count'].mean()
            stats['median_authors_per_paper'] = df['author_count'].median()
        
        # DOI coverage
        if 'has_doi' in df.columns:
            stats['papers_with_doi'] = df['has_doi'].sum()
            stats['doi_coverage_pct'] = (stats['papers_with_doi'] / len(df)) * 100
        
        return stats
    
    def get_collaboration_stats(self, df: pd.DataFrame) -> Dict:
        """Analyze collaboration patterns"""
        if 'author_count' not in df.columns:
            return {}
        
        stats = {
            'single_author_papers': (df['author_count'] == 1).sum(),
            'multi_author_papers': (df['author_count'] > 1).sum(),
            'avg_collaboration_size': df['author_count'].mean(),
            'max_authors': df['author_count'].max()
        }
        
        stats['collaboration_rate'] = (
            stats['multi_author_papers'] / len(df)
        ) * 100
        
        return stats
    
    def analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in publications"""
        if 'year' not in df.columns:
            return {}
        
        year_counts = df['year'].value_counts().sort_index()
        
        stats = {
            'most_productive_year': int(year_counts.idxmax()),
            'most_productive_year_count': int(year_counts.max()),
            'least_productive_year': int(year_counts.idxmin()),
            'least_productive_year_count': int(year_counts.min()),
            'total_years': len(year_counts)
        }
        
        # Calculate growth rate
        if len(year_counts) > 1:
            first_year_count = year_counts.iloc[0]
            last_year_count = year_counts.iloc[-1]
            stats['growth_rate'] = (
                (last_year_count - first_year_count) / first_year_count
            ) * 100
        
        return stats
    
    def get_journal_diversity(self, df: pd.DataFrame) -> Dict:
        """Calculate journal diversity metrics"""
        if 'journal' not in df.columns:
            return {}
        
        journal_counts = df['journal'].value_counts()
        total_papers = len(df)
        
        # Calculate concentration
        top_10_share = journal_counts.head(10).sum() / total_papers * 100
        top_20_share = journal_counts.head(20).sum() / total_papers * 100
        
        # Calculate Herfindahl index (concentration measure)
        proportions = journal_counts / total_papers
        herfindahl = (proportions ** 2).sum()
        
        return {
            'total_journals': len(journal_counts),
            'top_10_share_pct': top_10_share,
            'top_20_share_pct': top_20_share,
            'herfindahl_index': herfindahl,
            'effective_number_of_journals': 1 / herfindahl if herfindahl > 0 else 0
        }