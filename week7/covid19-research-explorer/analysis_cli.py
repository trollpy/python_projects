"""
COVID-19 Research Analysis - Command Line Interface
For batch analysis and report generation
"""

import argparse
import logging
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.analyzer import COVID19Analyzer
from src.visualizations import Visualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='COVID-19 Research Analysis CLI'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default=config.get('data.metadata_file'),
        help='Path to metadata.csv file'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Sample size (use None for full dataset)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for reports and visualizations'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'txt', 'html'],
        default='json',
        help='Output format for reports'
    )
    
    parser.add_argument(
        '--visualizations',
        action='store_true',
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--year-start',
        type=int,
        help='Filter start year'
    )
    
    parser.add_argument(
        '--year-end',
        type=int,
        help='Filter end year'
    )
    
    return parser.parse_args()


def generate_report(df, analyzer, output_dir, format='json'):
    """Generate comprehensive analysis report"""
    logger.info("Generating analysis report...")
    
    # Get all statistics
    stats = analyzer.get_summary_statistics(df)
    collab_stats = analyzer.get_collaboration_stats(df)
    temporal_stats = analyzer.analyze_temporal_patterns(df)
    journal_diversity = analyzer.get_journal_diversity(df)
    
    # Get top items
    top_journals = analyzer.get_top_journals(df, top_n=20)
    top_sources = analyzer.get_top_sources(df, top_n=10)
    publication_trends = analyzer.get_publication_trends(df)
    
    # Compile report
    report = {
        'metadata': {
            'generated_at': pd.Timestamp.now().isoformat(),
            'total_papers': len(df),
            'date_range': stats['date_range']
        },
        'summary_statistics': stats,
        'collaboration_statistics': collab_stats,
        'temporal_patterns': temporal_stats,
        'journal_diversity': journal_diversity,
        'top_journals': top_journals.to_dict(),
        'top_sources': top_sources.to_dict(),
        'publication_trends': publication_trends.to_dict()
    }
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save report based on format
    if format == 'json':
        report_file = output_path / 'analysis_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to {report_file}")
    
    elif format == 'txt':
        report_file = output_path / 'analysis_report.txt'
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COVID-19 RESEARCH ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {report['metadata']['generated_at']}\n")
            f.write(f"Total Papers: {report['metadata']['total_papers']:,}\n")
            f.write(f"Date Range: {report['metadata']['date_range']['min_year']}-"
                   f"{report['metadata']['date_range']['max_year']}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("-"*80 + "\n")
            for key, value in stats.items():
                if not isinstance(value, dict):
                    f.write(f"{key}: {value}\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("TOP 20 JOURNALS\n")
            f.write("-"*80 + "\n")
            for i, (journal, count) in enumerate(top_journals.items(), 1):
                f.write(f"{i}. {journal}: {count:,} papers\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("PUBLICATION TRENDS BY YEAR\n")
            f.write("-"*80 + "\n")
            for year, count in sorted(publication_trends.items()):
                f.write(f"{year}: {count:,} papers\n")
        
        logger.info(f"Report saved to {report_file}")
    
    elif format == 'html':
        report_file = output_path / 'analysis_report.html'
        html_content = generate_html_report(report, top_journals, publication_trends)
        with open(report_file, 'w') as f:
            f.write(html_content)
        logger.info(f"Report saved to {report_file}")
    
    return report


def generate_html_report(report, top_journals, publication_trends):
    """Generate HTML formatted report"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>COVID-19 Research Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .section {{
                background: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric {{
                display: inline-block;
                margin: 10px 20px 10px 0;
                padding: 15px;
                background: #e3f2fd;
                border-radius: 5px;
                border-left: 4px solid #2196f3;
            }}
            .metric-label {{
                font-size: 0.9rem;
                color: #666;
            }}
            .metric-value {{
                font-size: 1.5rem;
                font-weight: bold;
                color: #1976d2;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: 600;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            h2 {{
                color: #333;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🦠 COVID-19 Research Analysis Report</h1>
            <p>Generated: {report['metadata']['generated_at']}</p>
        </div>
        
        <div class="section">
            <h2>Overview</h2>
            <div class="metric">
                <div class="metric-label">Total Papers</div>
                <div class="metric-value">{report['metadata']['total_papers']:,}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Unique Journals</div>
                <div class="metric-value">{report['summary_statistics']['unique_journals']:,}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Year Range</div>
                <div class="metric-value">{report['metadata']['date_range']['min_year']}-{report['metadata']['date_range']['max_year']}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>Top 20 Journals</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Journal</th>
                        <th>Papers</th>
                        <th>Share (%)</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    total_papers = report['metadata']['total_papers']
    for i, (journal, count) in enumerate(list(top_journals.items())[:20], 1):
        share = (count / total_papers) * 100
        html += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{journal}</td>
                        <td>{count:,}</td>
                        <td>{share:.2f}%</td>
                    </tr>
        """
    
    html += """
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>Publication Trends</h2>
            <table>
                <thead>
                    <tr>
                        <th>Year</th>
                        <th>Publications</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for year, count in sorted(publication_trends.items()):
        html += f"""
                    <tr>
                        <td>{year}</td>
                        <td>{count:,}</td>
                    </tr>
        """
    
    html += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    return html


def generate_visualizations(df, analyzer, visualizer, output_dir):
    """Generate and save all visualizations"""
    logger.info("Generating visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Publications by year
    year_counts = analyzer.get_publication_trends(df)
    if not year_counts.empty:
        fig = visualizer.plot_publications_by_year(year_counts)
        fig.write_html(output_path / 'publications_by_year.html')
        fig.write_image(output_path / 'publications_by_year.png')
        logger.info("Generated: publications_by_year")
    
    # Top journals
    journal_counts = analyzer.get_top_journals(df, top_n=20)
    if not journal_counts.empty:
        fig = visualizer.plot_top_journals(journal_counts)
        fig.write_html(output_path / 'top_journals.html')
        fig.write_image(output_path / 'top_journals.png')
        logger.info("Generated: top_journals")
    
    # Source distribution
    source_counts = analyzer.get_top_sources(df, top_n=10)
    if not source_counts.empty:
        fig = visualizer.plot_source_distribution(source_counts)
        fig.write_html(output_path / 'source_distribution.html')
        fig.write_image(output_path / 'source_distribution.png')
        logger.info("Generated: source_distribution")
    
    # Word cloud
    try:
        fig = visualizer.create_wordcloud(df['title'])
        fig.savefig(output_path / 'wordcloud.png', dpi=300, bbox_inches='tight')
        logger.info("Generated: wordcloud")
    except Exception as e:
        logger.error(f"Error generating wordcloud: {e}")
    
    # Common words
    word_counts = analyzer.get_common_words(df['title'], top_n=30)
    if word_counts:
        fig = visualizer.plot_common_words(word_counts)
        fig.write_html(output_path / 'common_words.html')
        fig.write_image(output_path / 'common_words.png')
        logger.info("Generated: common_words")
    
    logger.info(f"All visualizations saved to {output_path}")


def main():
    """Main CLI execution"""
    args = parse_arguments()
    
    logger.info("="*80)
    logger.info("COVID-19 Research Analysis CLI")
    logger.info("="*80)
    
    try:
        # Load data
        logger.info(f"Loading data from: {args.data}")
        loader = DataLoader(args.data)
        df = loader.load_full(sample_size=args.sample)
        logger.info(f"Loaded {len(df):,} papers")
        
        # Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(df)
        logger.info(f"Cleaned data: {len(df_clean):,} papers")
        
        # Apply year filters if specified
        if args.year_start or args.year_end:
            logger.info(f"Applying year filter: {args.year_start} - {args.year_end}")
            df_clean = preprocessor.filter_by_date_range(
                df_clean,
                start_year=args.year_start,
                end_year=args.year_end
            )
            logger.info(f"Filtered data: {len(df_clean):,} papers")
        
        # Initialize analyzer and visualizer
        analyzer = COVID19Analyzer(
            stop_words=config.get('analysis.stop_words', [])
        )
        visualizer = Visualizer(config.get('visualization', {}))
        
        # Generate report
        report = generate_report(df_clean, analyzer, args.output, args.format)
        
        # Generate visualizations if requested
        if args.visualizations:
            generate_visualizations(df_clean, analyzer, visualizer, args.output)
        
        logger.info("="*80)
        logger.info("Analysis complete!")
        logger.info(f"Results saved to: {args.output}")
        logger.info("="*80)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please download the dataset from:")
        logger.error("https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()