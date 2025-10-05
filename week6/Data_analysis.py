"""
Post-COVID Conditions Data Analysis
A comprehensive analysis script covering data loading, exploration, analysis, and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import gaussian_kde
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def load_dataset(filepath):
    """
    Load the CSV dataset with error handling.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame or None if loading fails
    """
    try:
        print("=" * 70)
        print("TASK 1: LOADING AND EXPLORING THE DATASET")
        print("=" * 70)
        
        # Check if file exists
        if not Path(filepath).exists():
            print(f"Error: File '{filepath}' not found.")
            print("Please ensure the file is in the same directory as this script.")
            return None
        
        # Load the dataset
        print(f"\nLoading dataset from: {filepath}")
        df = pd.read_csv(filepath)
        print(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
        
        return df
        
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: Unable to parse the CSV file. Check the file format.")
        return None
    except Exception as e:
        print(f"Unexpected error loading dataset: {e}")
        return None


def explore_dataset(df):
    """
    Explore the structure and content of the dataset.
    
    Args:
        df: pandas DataFrame
    """
    print("\n" + "-" * 70)
    print("Dataset Preview (First 5 Rows)")
    print("-" * 70)
    print(df.head())
    
    print("\n" + "-" * 70)
    print("Dataset Information")
    print("-" * 70)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn Names:\n{df.columns.tolist()}")
    
    print("\n" + "-" * 70)
    print("Data Types")
    print("-" * 70)
    print(df.dtypes)
    
    print("\n" + "-" * 70)
    print("Missing Values Analysis")
    print("-" * 70)
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_percent
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    if missing.sum() == 0:
        print("No missing values found in the dataset.")


def clean_dataset(df):
    """
    Clean the dataset by handling missing values.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    print("\n" + "-" * 70)
    print("Data Cleaning")
    print("-" * 70)
    
    df_cleaned = df.copy()
    missing_before = df_cleaned.isnull().sum().sum()
    
    if missing_before > 0:
        print(f"Total missing values before cleaning: {missing_before}")
        
        # Strategy: Fill numerical columns with median, categorical with mode
        for column in df_cleaned.columns:
            if df_cleaned[column].isnull().sum() > 0:
                if df_cleaned[column].dtype in ['int64', 'float64']:
                    # Fill numerical columns with median
                    median_val = df_cleaned[column].median()
                    df_cleaned[column].fillna(median_val, inplace=True)
                    print(f"Filled '{column}' with median: {median_val}")
                else:
                    # Fill categorical columns with mode
                    mode_val = df_cleaned[column].mode()[0] if not df_cleaned[column].mode().empty else 'Unknown'
                    df_cleaned[column].fillna(mode_val, inplace=True)
                    print(f"Filled '{column}' with mode: {mode_val}")
        
        missing_after = df_cleaned.isnull().sum().sum()
        print(f"\nTotal missing values after cleaning: {missing_after}")
    else:
        print("No missing values to clean.")
    
    return df_cleaned


def analyze_dataset(df):
    """
    Perform basic statistical analysis on the dataset.
    
    Args:
        df: pandas DataFrame
    """
    print("\n" + "=" * 70)
    print("TASK 2: BASIC DATA ANALYSIS")
    print("=" * 70)
    
    # Basic statistics for numerical columns
    print("\n" + "-" * 70)
    print("Statistical Summary of Numerical Columns")
    print("-" * 70)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) > 0:
        print(df[numerical_cols].describe())
    else:
        print("No numerical columns found in the dataset.")
    
    # Group analysis on categorical columns
    print("\n" + "-" * 70)
    print("Group Analysis")
    print("-" * 70)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0 and len(numerical_cols) > 0:
        # Take first categorical and first numerical column for grouping
        cat_col = categorical_cols[0]
        num_col = numerical_cols[0]
        
        print(f"\nGrouping by '{cat_col}' and computing mean of '{num_col}':")
        grouped = df.groupby(cat_col)[num_col].agg(['mean', 'count', 'std'])
        print(grouped.sort_values('mean', ascending=False))
        
        # Additional insights
        print(f"\n\nKey Findings:")
        print(f"- Total unique categories in '{cat_col}': {df[cat_col].nunique()}")
        print(f"- Category with highest average {num_col}: {grouped['mean'].idxmax()}")
        print(f"- Category with lowest average {num_col}: {grouped['mean'].idxmin()}")
    else:
        print("Insufficient columns for grouping analysis.")


def visualize_data(df):
    """
    Create professional, publication-quality visualizations of the dataset.
    Each visualization is saved as a separate PNG file with elegant styling.
    
    Args:
        df: pandas DataFrame
    """
    print("\n" + "=" * 70)
    print("TASK 3: DATA VISUALIZATION")
    print("=" * 70)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Professional color palette
    colors = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'accent': '#F18F01',
        'success': '#06A77D',
        'danger': '#D00000',
        'neutral': '#6C757D'
    }
    
    print("\nCreating professional visualizations...")
    
    # 1. Line Chart - Temporal Trend Analysis
    print("1. Temporal Trend Analysis")
    
    if len(numerical_cols) >= 1:
        fig1, ax1 = plt.subplots(figsize=(14, 7))
        fig1.patch.set_facecolor('white')
        
        num_col = numerical_cols[0]
        data_series = df[num_col].dropna()
        
        # Plot with gradient effect
        ax1.plot(data_series.index, data_series.values, 
                color=colors['primary'], linewidth=2.5, alpha=0.9, label=num_col)
        ax1.fill_between(data_series.index, data_series.values, 
                         alpha=0.2, color=colors['primary'])
        
        # Add moving average if enough data
        if len(data_series) > 20:
            window = min(30, len(data_series) // 10)
            rolling_mean = data_series.rolling(window=window, center=True).mean()
            ax1.plot(rolling_mean.index, rolling_mean.values, 
                    color=colors['accent'], linewidth=2, linestyle='--', 
                    alpha=0.8, label=f'{window}-Record Moving Average')
        
        # Formatting
        ax1.set_title(f'Temporal Progression of {num_col}', 
                     fontsize=18, fontweight='bold', pad=20, color='#2C3E50')
        ax1.set_xlabel('Record Index', fontsize=13, fontweight='600', color='#34495E')
        ax1.set_ylabel(num_col, fontsize=13, fontweight='600', color='#34495E')
        
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_linewidth(1.5)
        ax1.spines['bottom'].set_linewidth(1.5)
        ax1.grid(True, alpha=0.15, linestyle='-', linewidth=0.8)
        ax1.legend(frameon=True, shadow=True, fontsize=11, loc='best')
        
        plt.tight_layout()
        plt.savefig('1_temporal_trend_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("   Saved: 1_temporal_trend_analysis.png")
        plt.close()
    
    # 2. Bar Chart - Categorical Comparison with Ranking
    print("2. Categorical Comparison Analysis")
    
    if len(categorical_cols) >= 1 and len(numerical_cols) >= 1:
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        fig2.patch.set_facecolor('white')
        
        cat_col = categorical_cols[0]
        num_col = numerical_cols[0]
        
        # Calculate statistics
        grouped = df.groupby(cat_col)[num_col].agg(['mean', 'count']).sort_values('mean', ascending=False)
        
        # Filter to top categories with sufficient data
        grouped_filtered = grouped[grouped['count'] >= 5].head(12)
        
        if len(grouped_filtered) > 0:
            # Create gradient colors
            n_bars = len(grouped_filtered)
            colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, n_bars))
            
            bars = ax2.barh(range(len(grouped_filtered)), grouped_filtered['mean'],
                           color=colors_gradient, edgecolor='white', linewidth=1.5)
            
            # Add value labels
            for i, (idx, row) in enumerate(grouped_filtered.iterrows()):
                ax2.text(row['mean'], i, f"  {row['mean']:.1f} (n={int(row['count'])})",
                        va='center', fontsize=10, fontweight='500', color='#2C3E50')
            
            # Formatting
            ax2.set_yticks(range(len(grouped_filtered)))
            ax2.set_yticklabels(grouped_filtered.index, fontsize=11, fontweight='500')
            ax2.set_xlabel(f'Mean {num_col}', fontsize=13, fontweight='600', color='#34495E')
            ax2.set_title(f'{num_col} Distribution Across {cat_col} Categories', 
                         fontsize=18, fontweight='bold', pad=20, color='#2C3E50')
            
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_linewidth(1.5)
            ax2.spines['bottom'].set_linewidth(1.5)
            ax2.grid(axis='x', alpha=0.15, linestyle='-', linewidth=0.8)
            ax2.invert_yaxis()
            
            plt.tight_layout()
            plt.savefig('2_categorical_comparison_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
            print("   Saved: 2_categorical_comparison_analysis.png")
        plt.close()
    
    # 3. Distribution Analysis with Statistics
    print("3. Distribution Analysis")
    
    if len(numerical_cols) >= 1:
        fig3, ax3 = plt.subplots(figsize=(14, 7))
        fig3.patch.set_facecolor('white')
        
        num_col = numerical_cols[0]
        data = df[num_col].dropna()
        
        # Create histogram
        n, bins, patches = ax3.hist(data, bins=40, color=colors['primary'], 
                                     alpha=0.7, edgecolor='white', linewidth=1.2)
        
        # Add KDE overlay
        from scipy import stats
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        kde_values = kde(x_range)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(x_range, kde_values, color=colors['accent'], 
                     linewidth=3, label='Density Curve')
        ax3_twin.fill_between(x_range, kde_values, alpha=0.15, color=colors['accent'])
        
        # Add statistical markers
        mean_val = data.mean()
        median_val = data.median()
        
        ax3.axvline(mean_val, color=colors['danger'], linestyle='--', 
                   linewidth=2.5, alpha=0.8, label=f'Mean: {mean_val:.2f}')
        ax3.axvline(median_val, color=colors['success'], linestyle='--', 
                   linewidth=2.5, alpha=0.8, label=f'Median: {median_val:.2f}')
        
        # Statistics box
        textstr = f'Statistics:\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd Dev: {data.std():.2f}\nN: {len(data):,}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5)
        ax3.text(0.98, 0.97, textstr, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right', bbox=props, fontweight='500')
        
        # Formatting
        ax3.set_xlabel(num_col, fontsize=13, fontweight='600', color='#34495E')
        ax3.set_ylabel('Frequency', fontsize=13, fontweight='600', color='#34495E')
        ax3_twin.set_ylabel('Probability Density', fontsize=13, fontweight='600', color='#34495E')
        ax3.set_title(f'Distribution Analysis of {num_col}', 
                     fontsize=18, fontweight='bold', pad=20, color='#2C3E50')
        
        ax3.spines['top'].set_visible(False)
        ax3_twin.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.legend(loc='upper left', frameon=True, shadow=True, fontsize=11)
        ax3_twin.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
        ax3.grid(axis='y', alpha=0.15, linestyle='-', linewidth=0.8)
        
        plt.tight_layout()
        plt.savefig('3_distribution_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("   Saved: 3_distribution_analysis.png")
        plt.close()
    
    # 4. Correlation Analysis with Regression
    print("4. Correlation Analysis")
    
    if len(numerical_cols) >= 2:
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        fig4.patch.set_facecolor('white')
        
        x_col = numerical_cols[0]
        y_col = numerical_cols[1]
        
        # Remove NaN values
        plot_data = df[[x_col, y_col]].dropna()
        
        # Create scatter with color gradient based on density
        from scipy.stats import gaussian_kde
        xy = np.vstack([plot_data[x_col], plot_data[y_col]])
        z = gaussian_kde(xy)(xy)
        
        scatter = ax4.scatter(plot_data[x_col], plot_data[y_col], 
                            c=z, s=50, alpha=0.6, cmap='viridis', 
                            edgecolors='white', linewidth=0.5)
        
        # Add regression line
        z_fit = np.polyfit(plot_data[x_col], plot_data[y_col], 1)
        p_fit = np.poly1d(z_fit)
        x_line = np.linspace(plot_data[x_col].min(), plot_data[x_col].max(), 100)
        ax4.plot(x_line, p_fit(x_line), color=colors['danger'], 
                linewidth=3, linestyle='--', alpha=0.8, label='Regression Line')
        
        # Calculate correlation
        corr = plot_data[x_col].corr(plot_data[y_col])
        
        # Correlation interpretation
        if abs(corr) > 0.7:
            strength = "Strong"
        elif abs(corr) > 0.4:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        direction = "Positive" if corr > 0 else "Negative"
        
        # Statistics box
        textstr = f'Correlation Analysis:\n\nPearson r: {corr:.3f}\nR²: {corr**2:.3f}\n\nInterpretation:\n{strength} {direction}\nCorrelation\n\nN: {len(plot_data):,} observations'
        props = dict(boxstyle='round', facecolor='white', alpha=0.95, 
                    edgecolor='gray', linewidth=2)
        ax4.text(0.02, 0.98, textstr, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', bbox=props, fontweight='500')
        
        # Formatting
        ax4.set_xlabel(x_col, fontsize=13, fontweight='600', color='#34495E')
        ax4.set_ylabel(y_col, fontsize=13, fontweight='600', color='#34495E')
        ax4.set_title(f'Correlation Analysis: {x_col} vs {y_col}', 
                     fontsize=18, fontweight='bold', pad=20, color='#2C3E50')
        
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_linewidth(1.5)
        ax4.spines['bottom'].set_linewidth(1.5)
        ax4.grid(True, alpha=0.15, linestyle='-', linewidth=0.8)
        ax4.legend(frameon=True, shadow=True, fontsize=11, loc='lower right')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4, pad=0.02)
        cbar.set_label('Point Density', fontsize=11, fontweight='500')
        
        plt.tight_layout()
        plt.savefig('4_correlation_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("   Saved: 4_correlation_analysis.png")
        plt.close()
    
    print("\nAll professional visualizations saved successfully!")


def generate_summary_report(df):
    """
    Generate a summary report of the analysis.
    
    Args:
        df: pandas DataFrame
    """
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    print(f"\nDataset Overview:")
    print(f"- Total Records: {len(df)}")
    print(f"- Total Columns: {len(df.columns)}")
    print(f"- Numerical Columns: {len(numerical_cols)}")
    print(f"- Categorical Columns: {len(categorical_cols)}")
    
    if len(numerical_cols) > 0:
        print(f"\nNumerical Data Insights:")
        for col in numerical_cols[:3]:  # Show first 3 numerical columns
            print(f"- {col}:")
            print(f"  Mean: {df[col].mean():.2f}")
            print(f"  Median: {df[col].median():.2f}")
            print(f"  Std Dev: {df[col].std():.2f}")
    
    if len(categorical_cols) > 0:
        print(f"\nCategorical Data Insights:")
        for col in categorical_cols[:3]:  # Show first 3 categorical columns
            print(f"- {col}: {df[col].nunique()} unique values")
            print(f"  Most common: {df[col].mode()[0] if not df[col].mode().empty else 'N/A'}")
    
    print("\n" + "=" * 70)


def main():
    """Main function to orchestrate the analysis."""
    
    # File path - adjust this to your CSV filename
    filepath = "Post-COVID_Conditions.csv"
    
    try:
        # Task 1: Load and Explore
        df = load_dataset(filepath)
        
        if df is None:
            print("\nAnalysis terminated due to loading error.")
            return
        
        explore_dataset(df)
        df_cleaned = clean_dataset(df)
        
        # Task 2: Basic Analysis
        analyze_dataset(df_cleaned)
        
        # Task 3: Visualization
        visualize_data(df_cleaned)
        
        # Generate summary
        generate_summary_report(df_cleaned)
        
        print("\nAnalysis completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()