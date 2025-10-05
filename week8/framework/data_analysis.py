"""
COVID-19 Data Analysis Script
This script performs exploratory data analysis on the COVID-19 metadata dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class COVID19Analyzer:
    """Class to handle COVID-19 dataset analysis"""
    
    def __init__(self, filepath):
        """Initialize analyzer with dataset filepath"""
        self.filepath = filepath
        self.df = None
        self.df_clean = None
        
    def load_data(self, sample_size=None):
        """Load the metadata CSV file"""
        print("Loading data...")
        if sample_size:
            self.df = pd.read_csv(self.filepath, nrows=sample_size)
            print(f"Loaded sample of {sample_size} rows")
        else:
            self.df = pd.read_csv(self.filepath)
        
        print(f"Dataset shape: {self.df.shape}")
        return self.df
    
    def explore_data(self):
        """Perform basic data exploration"""
        print("\n=== DATA EXPLORATION ===")
        print(f"\nDataset dimensions: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        print("\nColumn names:")
        print(self.df.columns.tolist())
        
        print("\nData types:")
        print(self.df.dtypes)
        
        print("\nFirst few rows:")
        print(self.df.head())
        
        print("\nBasic statistics:")
        print(self.df.describe())
        
        print("\nMissing values:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0].sort_values('Percentage', ascending=False))
        
    def clean_data(self):
        """Clean and prepare the dataset"""
        print("\n=== DATA CLEANING ===")
        self.df_clean = self.df.copy()
        
        # Convert publish_time to datetime
        print("Converting dates...")
        self.df_clean['publish_time'] = pd.to_datetime(
            self.df_clean['publish_time'], 
            errors='coerce'
        )
        
        # Extract year
        self.df_clean['year'] = self.df_clean['publish_time'].dt.year
        
        # Remove rows with no title or abstract
        print("Removing rows with missing titles...")
        initial_rows = len(self.df_clean)
        self.df_clean = self.df_clean.dropna(subset=['title'])
        print(f"Removed {initial_rows - len(self.df_clean)} rows")
        
        # Create abstract word count
        print("Calculating abstract word counts...")
        self.df_clean['abstract_word_count'] = self.df_clean['abstract'].fillna('').apply(
            lambda x: len(str(x).split())
        )
        
        # Create title word count
        self.df_clean['title_word_count'] = self.df_clean['title'].fillna('').apply(
            lambda x: len(str(x).split())
        )
        
        print(f"Cleaned dataset shape: {self.df_clean.shape}")
        return self.df_clean
    
    def analyze_publications_by_year(self):
        """Analyze publication trends over time"""
        print("\n=== PUBLICATIONS BY YEAR ===")
        year_counts = self.df_clean['year'].value_counts().sort_index()
        print(year_counts)
        return year_counts
    
    def analyze_top_journals(self, top_n=10):
        """Find top journals publishing COVID-19 research"""
        print(f"\n=== TOP {top_n} JOURNALS ===")
        journal_counts = self.df_clean['journal'].value_counts().head(top_n)
        print(journal_counts)
        return journal_counts
    
    def analyze_top_sources(self, top_n=10):
        """Find top sources of papers"""
        print(f"\n=== TOP {top_n} SOURCES ===")
        source_counts = self.df_clean['source_x'].value_counts().head(top_n)
        print(source_counts)
        return source_counts
    
    def get_common_words(self, text_series, top_n=20):
        """Extract most common words from text"""
        # Combine all text
        all_text = ' '.join(text_series.dropna().astype(str).tolist())
        
        # Clean and tokenize
        words = re.findall(r'\b[a-z]{4,}\b', all_text.lower())
        
        # Remove common stop words
        stop_words = {'with', 'from', 'that', 'this', 'have', 'been', 'were', 
                     'their', 'which', 'about', 'there', 'these', 'would'}
        words = [w for w in words if w not in stop_words]
        
        # Count words
        word_counts = Counter(words)
        return word_counts.most_common(top_n)
    
    def plot_publications_by_year(self, save_path=None):
        """Create bar plot of publications by year"""
        year_counts = self.df_clean['year'].value_counts().sort_index()
        
        plt.figure(figsize=(12, 6))
        plt.bar(year_counts.index, year_counts.values, color='steelblue', edgecolor='black')
        plt.title('COVID-19 Research Publications by Year', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Number of Publications', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_top_journals(self, top_n=15, save_path=None):
        """Create horizontal bar plot of top journals"""
        journal_counts = self.df_clean['journal'].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(journal_counts)), journal_counts.values, color='coral')
        plt.yticks(range(len(journal_counts)), journal_counts.index)
        plt.title(f'Top {top_n} Journals Publishing COVID-19 Research', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Number of Publications', fontsize=12)
        plt.ylabel('Journal', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_wordcloud(self, save_path=None):
        """Create word cloud from paper titles"""
        all_titles = ' '.join(self.df_clean['title'].dropna().astype(str).tolist())
        
        wordcloud = WordCloud(
            width=1200, 
            height=600, 
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(all_titles)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Paper Titles', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_source_distribution(self, top_n=10, save_path=None):
        """Plot distribution of papers by source"""
        source_counts = self.df_clean['source_x'].value_counts().head(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%',
                startangle=90, colors=sns.color_palette('Set3'))
        plt.title(f'Distribution of Papers by Source (Top {top_n})', 
                 fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate summary report"""
        print("\n" + "="*60)
        print("COVID-19 DATA ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nTotal papers analyzed: {len(self.df_clean)}")
        print(f"Date range: {self.df_clean['year'].min():.0f} - {self.df_clean['year'].max():.0f}")
        print(f"Unique journals: {self.df_clean['journal'].nunique()}")
        print(f"Average abstract length: {self.df_clean['abstract_word_count'].mean():.1f} words")
        
        print("\nKey Findings:")
        print(f"- Most productive year: {self.df_clean['year'].mode()[0]:.0f}")
        print(f"- Top journal: {self.df_clean['journal'].mode()[0]}")
        
        print("\n" + "="*60)


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = COVID19Analyzer('metadata.csv')
    
    # Load data (use sample_size parameter for large files)
    analyzer.load_data(sample_size=10000)  # Adjust or remove sample_size as needed
    
    # Explore data
    analyzer.explore_data()
    
    # Clean data
    analyzer.clean_data()
    
    # Perform analysis
    analyzer.analyze_publications_by_year()
    analyzer.analyze_top_journals()
    analyzer.analyze_top_sources()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    analyzer.plot_publications_by_year(save_path='publications_by_year.png')
    analyzer.plot_top_journals(save_path='top_journals.png')
    analyzer.plot_wordcloud(save_path='title_wordcloud.png')
    analyzer.plot_source_distribution(save_path='source_distribution.png')
    
    # Generate report
    analyzer.generate_report()
    
    print("\nAnalysis complete!")