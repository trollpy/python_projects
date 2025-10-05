"""
COVID-19 Research Explorer - Streamlit Application
Interactive dashboard for exploring COVID-19 research papers
"""

import streamlit as st
import pandas as pd
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.analyzer import COVID19Analyzer
from src.visualizations import Visualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=config.get('streamlit.page_title', 'COVID-19 Research Explorer'),
    page_icon=config.get('streamlit.page_icon', '🦠'),
    layout=config.get('streamlit.layout', 'wide'),
    initial_sidebar_state=config.get('streamlit.initial_sidebar_state', 'expanded')
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    .info-box {
        padding: 1rem;
        background-color: #e3f2fd;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3e0;
        border-radius: 5px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data(filepath: str, sample_size: int = None):
    """Load and cache the dataset"""
    try:
        loader = DataLoader(filepath)
        df = loader.load_full(sample_size=sample_size)
        return df, None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, f"Error loading data: {str(e)}"


@st.cache_data(show_spinner=False)
def preprocess_data(_df):
    """Preprocess and cache the cleaned dataset"""
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(_df)
    return df_clean, preprocessor.cleaning_stats


def show_data_not_found_message():
    """Display message when data file is not found"""
    st.error("📁 Data file not found!")
    
    st.markdown("""
    <div class="warning-box">
    <h3>⚠️ Setup Required</h3>
    <p>Please download the COVID-19 dataset to continue:</p>
    <ol>
        <li>Visit <a href="https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge" target="_blank">Kaggle CORD-19 Dataset</a></li>
        <li>Download the <code>metadata.csv</code> file</li>
        <li>Place it in the <code>data/</code> directory</li>
        <li>Refresh this page</li>
    </ol>
    <p><strong>Expected file location:</strong> <code>data/metadata.csv</code></p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🔧 Alternative: Use Custom Path"):
        custom_path = st.text_input("Enter custom data path:", value="data/metadata.csv")
        if st.button("Load from Custom Path"):
            st.session_state.custom_data_path = custom_path
            st.rerun()


def render_header():
    """Render application header"""
    st.markdown('<h1 class="main-header">🦠 COVID-19 Research Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive analysis of COVID-19 research publications</p>', unsafe_allow_html=True)
    st.markdown("---")


def render_sidebar(df_clean):
    """Render sidebar with filters and configuration"""
    st.sidebar.header("⚙️ Configuration")
    
    # Data loading section
    with st.sidebar.expander("📊 Data Settings", expanded=False):
        sample_size = st.number_input(
            "Sample size (0 = all data)",
            min_value=0,
            max_value=1000000,
            value=config.get('data.sample_size', 50000),
            step=10000,
            help="Reduce for faster loading. Use 0 to load full dataset."
        )
        
        if st.button("🔄 Reload Data"):
            st.cache_data.clear()
            st.rerun()
    
    st.sidebar.success(f"✅ Loaded {len(df_clean):,} papers")
    
    # Filters section
    st.sidebar.header("🔍 Filters")
    
    # Year range filter
    min_year = int(df_clean['year'].min())
    max_year = int(df_clean['year'].max())
    
    year_range = st.sidebar.slider(
        "📅 Year Range",
        min_year,
        max_year,
        (min_year, max_year),
        help="Filter papers by publication year"
    )
    
    # Source filter
    st.sidebar.subheader("📚 Sources")
    top_sources = df_clean['source_x'].value_counts().head(15).index.tolist()
    
    source_options = st.sidebar.multiselect(
        "Select sources to include",
        options=top_sources,
        default=top_sources[:5] if len(top_sources) >= 5 else top_sources,
        help="Filter by data source"
    )
    
    # Journal filter
    st.sidebar.subheader("📖 Journals")
    show_journal_filter = st.sidebar.checkbox("Enable journal filter", value=False)
    
    selected_journals = []
    if show_journal_filter:
        top_journals = df_clean['journal'].value_counts().head(20).index.tolist()
        selected_journals = st.sidebar.multiselect(
            "Select journals",
            options=top_journals,
            help="Filter by journal"
        )
    
    # Abstract filter
    st.sidebar.subheader("📝 Content Filters")
    only_with_abstract = st.sidebar.checkbox(
        "Only papers with abstracts",
        value=False,
        help="Show only papers that have abstracts"
    )
    
    min_abstract_length = st.sidebar.slider(
        "Minimum abstract length (words)",
        0,
        500,
        0,
        help="Filter by minimum abstract word count"
    )
    
    return {
        'sample_size': sample_size if sample_size > 0 else None,
        'year_range': year_range,
        'sources': source_options,
        'journals': selected_journals if show_journal_filter else None,
        'only_with_abstract': only_with_abstract,
        'min_abstract_length': min_abstract_length
    }


def apply_filters(df, filters):
    """Apply selected filters to dataframe"""
    df_filtered = df.copy()
    
    # Year filter
    df_filtered = df_filtered[
        (df_filtered['year'] >= filters['year_range'][0]) &
        (df_filtered['year'] <= filters['year_range'][1])
    ]
    
    # Source filter
    if filters['sources']:
        df_filtered = df_filtered[df_filtered['source_x'].isin(filters['sources'])]
    
    # Journal filter
    if filters['journals']:
        df_filtered = df_filtered[df_filtered['journal'].isin(filters['journals'])]
    
    # Abstract filters
    if filters['only_with_abstract']:
        df_filtered = df_filtered[df_filtered['has_abstract'] == True]
    
    if filters['min_abstract_length'] > 0:
        df_filtered = df_filtered[
            df_filtered['abstract_word_count'] >= filters['min_abstract_length']
        ]
    
    return df_filtered


def render_overview_tab(df_filtered, analyzer):
    """Render overview tab"""
    st.header("📊 Dataset Overview")
    
    # Get statistics
    stats = analyzer.get_summary_statistics(df_filtered)
    collab_stats = analyzer.get_collaboration_stats(df_filtered)
    temporal_stats = analyzer.analyze_temporal_patterns(df_filtered)
    journal_diversity = analyzer.get_journal_diversity(df_filtered)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📄 Total Papers",
            f"{stats['total_papers']:,}",
            help="Total number of papers in filtered dataset"
        )
    
    with col2:
        st.metric(
            "📚 Unique Journals",
            f"{stats['unique_journals']:,}",
            help="Number of unique journals"
        )
    
    with col3:
        if stats.get('avg_abstract_length'):
            st.metric(
                "📝 Avg Abstract Length",
                f"{stats['avg_abstract_length']:.0f} words",
                help="Average abstract word count"
            )
    
    with col4:
        year_range = f"{stats['date_range']['min_year']}-{stats['date_range']['max_year']}"
        st.metric(
            "📅 Year Range",
            year_range,
            help="Publication year range"
        )
    
    st.markdown("---")
    
    # Detailed statistics in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Publication Statistics")
        
        if temporal_stats:
            st.markdown(f"""
            <div class="info-box">
            <strong>Most Productive Year:</strong> {temporal_stats['most_productive_year']} 
            ({temporal_stats['most_productive_year_count']:,} papers)<br>
            <strong>Total Years Covered:</strong> {temporal_stats['total_years']}<br>
            <strong>Growth Rate:</strong> {temporal_stats.get('growth_rate', 0):.1f}%
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("👥 Collaboration Patterns")
        
        if collab_stats:
            st.markdown(f"""
            <div class="info-box">
            <strong>Multi-author Papers:</strong> {collab_stats['multi_author_papers']:,} 
            ({collab_stats['collaboration_rate']:.1f}%)<br>
            <strong>Avg Team Size:</strong> {collab_stats['avg_collaboration_size']:.1f} authors<br>
            <strong>Largest Team:</strong> {collab_stats['max_authors']} authors
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Content Coverage")
        
        if stats.get('papers_with_abstract'):
            st.markdown(f"""
            <div class="info-box">
            <strong>Papers with Abstracts:</strong> {stats['papers_with_abstract']:,} 
            ({stats['abstract_coverage_pct']:.1f}%)<br>
            <strong>Papers with DOI:</strong> {stats.get('papers_with_doi', 0):,} 
            ({stats.get('doi_coverage_pct', 0):.1f}%)
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("📖 Journal Diversity")
        
        if journal_diversity:
            st.markdown(f"""
            <div class="info-box">
            <strong>Total Journals:</strong> {journal_diversity['total_journals']:,}<br>
            <strong>Top 10 Share:</strong> {journal_diversity['top_10_share_pct']:.1f}%<br>
            <strong>Top 20 Share:</strong> {journal_diversity['top_20_share_pct']:.1f}%<br>
            <strong>Effective # Journals:</strong> {journal_diversity['effective_number_of_journals']:.0f}
            </div>
            """, unsafe_allow_html=True)


def render_time_analysis_tab(df_filtered, analyzer, visualizer):
    """Render time analysis tab"""
    st.header("📈 Publication Trends Over Time")
    
    # Year trends
    st.subheader("Annual Publication Trends")
    year_counts = analyzer.get_publication_trends(df_filtered)
    
    if not year_counts.empty:
        fig = visualizer.plot_publications_by_year(
            year_counts,
            title="COVID-19 Research Publications by Year"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        with st.expander("📊 View Data Table"):
            year_df = pd.DataFrame({
                'Year': year_counts.index,
                'Publications': year_counts.values,
                'Percentage': (year_counts.values / year_counts.sum() * 100).round(2)
            })
            st.dataframe(year_df, use_container_width=True)
    
    st.markdown("---")
    
    # Monthly trends
    st.subheader("Monthly Publication Trends")
    monthly_counts = analyzer.get_monthly_trends(df_filtered)
    
    if not monthly_counts.empty and len(monthly_counts) > 1:
        fig = visualizer.plot_monthly_trends(
            monthly_counts,
            title="Monthly Publication Trend"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Peak Month", str(monthly_counts.idxmax()))
        with col2:
            st.metric("Peak Publications", f"{monthly_counts.max():,}")
        with col3:
            st.metric("Avg per Month", f"{monthly_counts.mean():.0f}")
    else:
        st.info("Not enough data for monthly trend analysis")


def render_journals_sources_tab(df_filtered, analyzer, visualizer):
    """Render journals and sources tab"""
    st.header("📚 Journals and Sources Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Journals")
        
        top_n_journals = st.slider(
            "Number of journals to display",
            5, 30, 15,
            key='journals_slider'
        )
        
        journal_counts = analyzer.get_top_journals(df_filtered, top_n=top_n_journals)
        
        if not journal_counts.empty:
            fig = visualizer.plot_top_journals(
                journal_counts,
                title=f"Top {top_n_journals} Journals"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download button
            csv = journal_counts.to_csv()
            st.download_button(
                "📥 Download Journal Data",
                csv,
                "top_journals.csv",
                "text/csv",
                key='download-journals'
            )
    
    with col2:
        st.subheader("Source Distribution")
        
        top_n_sources = st.slider(
            "Number of sources to display",
            5, 15, 10,
            key='sources_slider'
        )
        
        source_counts = analyzer.get_top_sources(df_filtered, top_n=top_n_sources)
        
        if not source_counts.empty:
            fig = visualizer.plot_source_distribution(
                source_counts,
                title=f"Top {top_n_sources} Sources"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Source details
            with st.expander("📊 Source Statistics"):
                source_df = pd.DataFrame({
                    'Source': source_counts.index,
                    'Papers': source_counts.values,
                    'Share (%)': (source_counts.values / source_counts.sum() * 100).round(2)
                })
                st.dataframe(source_df, use_container_width=True)


def render_word_analysis_tab(df_filtered, analyzer, visualizer):
    """Render word analysis tab"""
    st.header("☁️ Text and Word Analysis")
    
    # Word cloud
    st.subheader("Word Cloud from Paper Titles")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        generate_wordcloud = st.button("🎨 Generate Word Cloud", use_container_width=True)
    
    if generate_wordcloud:
        with st.spinner("Generating word cloud..."):
            try:
                fig = visualizer.create_wordcloud(
                    df_filtered['title'],
                    title="Most Common Words in Paper Titles"
                )
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating word cloud: {e}")
    
    st.markdown("---")
    
    # Common words analysis
    st.subheader("Most Common Words in Titles")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        top_n_words = st.number_input(
            "Number of words",
            10, 100, 30,
            key='words_count'
        )
    
    word_counts = analyzer.get_common_words(
        df_filtered['title'],
        top_n=top_n_words,
        min_length=config.get('analysis.word_min_length', 4)
    )
    
    if word_counts:
        fig = visualizer.plot_common_words(
            word_counts,
            title=f"Top {top_n_words} Most Common Words"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Word table
        with st.expander("📋 Word Frequency Table"):
            words_df = pd.DataFrame(word_counts, columns=['Word', 'Frequency'])
            words_df['Rank'] = range(1, len(words_df) + 1)
            words_df = words_df[['Rank', 'Word', 'Frequency']]
            st.dataframe(words_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Additional text statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📏 Abstract Length Distribution")
        if 'abstract_word_count' in df_filtered.columns:
            fig = visualizer.plot_abstract_length_distribution(df_filtered)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("👥 Author Count Distribution")
        if 'author_count' in df_filtered.columns:
            fig = visualizer.plot_collaboration_distribution(df_filtered)
            st.plotly_chart(fig, use_container_width=True)


def render_data_explorer_tab(df_filtered):
    """Render data explorer tab"""
    st.header("🔍 Data Explorer")
    
    # Search functionality
    st.subheader("Search Papers")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_term = st.text_input(
            "Search in titles and abstracts",
            placeholder="Enter keywords...",
            key='search_input'
        )
    
    with col2:
        search_in = st.multiselect(
            "Search in",
            ['title', 'abstract', 'authors'],
            default=['title', 'abstract'],
            key='search_fields'
        )
    
    # Apply search
    if search_term:
        preprocessor = DataPreprocessor()
        display_df = preprocessor.search_text(df_filtered, search_term, columns=search_in)
        st.success(f"✅ Found {len(display_df):,} papers matching '{search_term}'")
    else:
        display_df = df_filtered
    
    st.markdown("---")
    
    # Column selection
    st.subheader("Select Columns to Display")
    
    available_columns = [
        'title', 'abstract', 'authors', 'journal', 
        'publish_time', 'source_x', 'doi', 'url',
        'year', 'abstract_word_count', 'author_count'
    ]
    
    # Filter to only existing columns
    available_columns = [col for col in available_columns if col in display_df.columns]
    
    default_columns = ['title', 'journal', 'publish_time', 'authors']
    default_columns = [col for col in default_columns if col in available_columns]
    
    selected_columns = st.multiselect(
        "Columns",
        available_columns,
        default=default_columns,
        key='display_columns'
    )
    
    # Display options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_rows = st.number_input(
            "Max rows to display",
            10, 1000, 100,
            key='max_rows'
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            selected_columns if selected_columns else ['title'],
            key='sort_by'
        )
    
    with col3:
        sort_order = st.radio(
            "Sort order",
            ['Ascending', 'Descending'],
            key='sort_order',
            horizontal=True
        )
    
    # Display data
    if selected_columns:
        display_subset = display_df[selected_columns].copy()
        
        # Sort
        ascending = sort_order == 'Ascending'
        display_subset = display_subset.sort_values(by=sort_by, ascending=ascending)
        
        # Show data
        st.dataframe(
            display_subset.head(max_rows),
            use_container_width=True,
            height=500
        )
        
        # Download options
        st.subheader("📥 Download Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV download
            csv = display_subset.head(max_rows).to_csv(index=False)
            st.download_button(
                "Download as CSV",
                csv,
                "covid19_filtered_data.csv",
                "text/csv",
                key='download-csv'
            )
        
        with col2:
            # JSON download
            json = display_subset.head(max_rows).to_json(orient='records', indent=2)
            st.download_button(
                "Download as JSON",
                json,
                "covid19_filtered_data.json",
                "application/json",
                key='download-json'
            )
        
        with col3:
            # Excel download
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                display_subset.head(max_rows).to_excel(writer, index=False, sheet_name='COVID-19 Data')
            
            st.download_button(
                "Download as Excel",
                buffer.getvalue(),
                "covid19_filtered_data.xlsx",
                "application/vnd.ms-excel",
                key='download-excel'
            )
    else:
        st.warning("⚠️ Please select at least one column to display")
    
    # Data statistics
    st.markdown("---")
    st.subheader("📊 Current View Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Papers", f"{len(display_df):,}")
    
    with col2:
        if 'journal' in display_df.columns:
            st.metric("Unique Journals", f"{display_df['journal'].nunique():,}")
    
    with col3:
        if 'year' in display_df.columns:
            st.metric("Year Range", f"{int(display_df['year'].min())}-{int(display_df['year'].max())}")
    
    with col4:
        if 'has_abstract' in display_df.columns:
            pct = (display_df['has_abstract'].sum() / len(display_df)) * 100
            st.metric("With Abstract", f"{pct:.1f}%")


def main():
    """Main application logic"""
    
    # Render header
    render_header()
    
    # Get data path
    data_path = st.session_state.get('custom_data_path', config.get('data.metadata_file'))
    
    # Load data
    with st.spinner("🔄 Loading dataset..."):
        df, error = load_data(data_path, sample_size=config.get('data.sample_size'))
    
    if df is None:
        show_data_not_found_message()
        st.stop()
    
    # Preprocess data
    with st.spinner("🧹 Cleaning and preparing data..."):
        df_clean, cleaning_stats = preprocess_data(df)
    
    # Initialize components
    analyzer = COVID19Analyzer(
        stop_words=config.get('analysis.stop_words', [])
    )
    visualizer = Visualizer(config.get('visualization', {}))
    
    # Render sidebar and get filters
    filters = render_sidebar(df_clean)
    
    # Apply filters
    df_filtered = apply_filters(df_clean, filters)
    
    # Show filter info
    if len(df_filtered) < len(df_clean):
        st.info(f"🔍 Filtered to {len(df_filtered):,} papers (from {len(df_clean):,} total)")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Time Analysis",
        "📚 Journals & Sources",
        "☁️ Word Analysis",
        "🔍 Data Explorer"
    ])
    
    with tab1:
        render_overview_tab(df_filtered, analyzer)
    
    with tab2:
        render_time_analysis_tab(df_filtered, analyzer, visualizer)
    
    with tab3:
        render_journals_sources_tab(df_filtered, analyzer, visualizer)
    
    with tab4:
        render_word_analysis_tab(df_filtered, analyzer, visualizer)
    
    with tab5:
        render_data_explorer_tab(df_filtered)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>COVID-19 Research Explorer | Data from 
        <a href="https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge" target="_blank">
        Kaggle CORD-19 Dataset</a></p>
        <p style="font-size: 0.9rem;">Built with Streamlit • Python • Plotly</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()