"""
COVID-19 Research Explorer - Streamlit Application
Interactive dashboard for exploring COVID-19 research papers
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

# Page configuration
st.set_page_config(
    page_title="COVID-19 Explorer",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(sample_size=None):
    """Load and cache the dataset"""
    try:
        if sample_size:
            df = pd.read_csv('metadata.csv', nrows=sample_size)
        else:
            df = pd.read_csv('metadata.csv')
        return df
    except FileNotFoundError:
        st.error("metadata.csv file not found. Please ensure the file is in the same directory as the app.")
        return None

@st.cache_data
def prepare_data(df):
    """Clean and prepare the dataset"""
    df_clean = df.copy()
    
    # Convert dates
    df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
    df_clean['year'] = df_clean['publish_time'].dt.year
    
    # Remove rows without title
    df_clean = df_clean.dropna(subset=['title'])
    
    # Calculate word counts
    df_clean['abstract_word_count'] = df_clean['abstract'].fillna('').apply(
        lambda x: len(str(x).split())
    )
    df_clean['title_word_count'] = df_clean['title'].fillna('').apply(
        lambda x: len(str(x).split())
    )
    
    return df_clean

def main():
    # Header
    st.markdown('<h1 class="main-header">🦠 COVID-19 Research Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive exploration of COVID-19 research papers</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Data loading options
    st.sidebar.subheader("Data Loading")
    sample_size = st.sidebar.number_input(
        "Sample size (0 for all data)",
        min_value=0,
        max_value=1000000,
        value=10000,
        step=1000,
        help="Use a smaller sample for faster loading"
    )
    
    if sample_size == 0:
        sample_size = None
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data(sample_size)
    
    if df is None:
        st.stop()
    
    # Prepare data
    with st.spinner("Preparing data..."):
        df_clean = prepare_data(df)
    
    # Display data info
    st.sidebar.success(f"✅ Loaded {len(df_clean):,} papers")
    
    # Sidebar filters
    st.sidebar.subheader("🔍 Filters")
    
    # Year range filter
    min_year = int(df_clean['year'].min())
    max_year = int(df_clean['year'].max())
    year_range = st.sidebar.slider(
        "Select year range",
        min_year,
        max_year,
        (min_year, max_year)
    )
    
    # Filter data by year
    df_filtered = df_clean[
        (df_clean['year'] >= year_range[0]) & 
        (df_clean['year'] <= year_range[1])
    ]
    
    # Source filter
    sources = df_filtered['source_x'].value_counts().head(10).index.tolist()
    selected_sources = st.sidebar.multiselect(
        "Select sources",
        options=sources,
        default=sources[:3] if len(sources) >= 3 else sources
    )
    
    if selected_sources:
        df_filtered = df_filtered[df_filtered['source_x'].isin(selected_sources)]
    
    st.sidebar.info(f"Filtered to {len(df_filtered):,} papers")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Time Analysis", 
        "📚 Journals & Sources",
        "☁️ Word Analysis",
        "📄 Data Explorer"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Papers", f"{len(df_filtered):,}")
        
        with col2:
            st.metric("Unique Journals", f"{df_filtered['journal'].nunique():,}")
        
        with col3:
            avg_abstract = df_filtered['abstract_word_count'].mean()
            st.metric("Avg Abstract Length", f"{avg_abstract:.0f} words")
        
        with col4:
            year_span = f"{int(df_filtered['year'].min())}-{int(df_filtered['year'].max())}"
            st.metric("Year Range", year_span)
        
        st.markdown("---")
        
        # Key statistics
        st.subheader("Key Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Most Productive Year:**")
            most_productive_year = df_filtered['year'].mode()[0]
            year_count = (df_filtered['year'] == most_productive_year).sum()
            st.info(f"{int(most_productive_year)} ({year_count:,} papers)")
            
            st.write("**Top Journal:**")
            top_journal = df_filtered['journal'].mode()[0] if len(df_filtered['journal'].mode()) > 0 else "N/A"
            st.info(f"{top_journal}")
        
        with col2:
            st.write("**Papers with Abstracts:**")
            papers_with_abstract = df_filtered['abstract'].notna().sum()
            pct = (papers_with_abstract / len(df_filtered)) * 100
            st.info(f"{papers_with_abstract:,} ({pct:.1f}%)")
            
            st.write("**Top Source:**")
            top_source = df_filtered['source_x'].mode()[0] if len(df_filtered['source_x'].mode()) > 0 else "N/A"
            st.info(f"{top_source}")
    
    # Tab 2: Time Analysis
    with tab2:
        st.header("Publication Trends Over Time")
        
        # Publications by year
        year_counts = df_filtered['year'].value_counts().sort_index()
        
        fig = px.bar(
            x=year_counts.index,
            y=year_counts.values,
            labels={'x': 'Year', 'y': 'Number of Publications'},
            title='COVID-19 Research Publications by Year'
        )
        fig.update_traces(marker_color='steelblue')
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly trends (if data available)
        st.subheader("Monthly Publication Trends")
        df_filtered['month'] = df_filtered['publish_time'].dt.to_period('M')
        monthly_counts = df_filtered['month'].value_counts().sort_index()
        
        if len(monthly_counts) > 0:
            monthly_df = pd.DataFrame({
                'Month': monthly_counts.index.astype(str),
                'Publications': monthly_counts.values
            })
            
            fig2 = px.line(
                monthly_df,
                x='Month',
                y='Publications',
                title='Monthly Publication Trend'
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # Tab 3: Journals & Sources
    with tab3:
        st.header("Journals and Sources Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Journals")
            top_n_journals = st.slider("Number of journals to show", 5, 20, 10, key='journals')
            journal_counts = df_filtered['journal'].value_counts().head(top_n_journals)
            
            fig = px.bar(
                x=journal_counts.values,
                y=journal_counts.index,
                orientation='h',
                labels={'x': 'Number of Publications', 'y': 'Journal'},
                title=f'Top {top_n_journals} Journals'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Source Distribution")
            source_counts = df_filtered['source_x'].value_counts().head(10)
            
            fig = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                title='Distribution by Source'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Word Analysis
    with tab4:
        st.header("Text Analysis")
        
        # Word cloud
        st.subheader("Word Cloud from Titles")
        
        if st.button("Generate Word Cloud"):
            with st.spinner("Generating word cloud..."):
                all_titles = ' '.join(df_filtered['title'].dropna().astype(str).tolist())
                
                wordcloud = WordCloud(
                    width=1200,
                    height=600,
                    background_color='white',
                    colormap='viridis',
                    max_words=100
                ).generate(all_titles)
                
                fig, ax = plt.subplots(figsize=(15, 8))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
        
        # Common words
        st.subheader("Most Common Words in Titles")
        top_n_words = st.slider("Number of words to show", 10, 50, 20, key='words')
        
        all_titles = ' '.join(df_filtered['title'].dropna().astype(str).tolist())
        words = re.findall(r'\b[a-z]{4,}\b', all_titles.lower())
        
        stop_words = {'with', 'from', 'that', 'this', 'have', 'been', 'were',
                     'their', 'which', 'about', 'there', 'these', 'would', 'among'}
        words = [w for w in words if w not in stop_words]
        
        word_counts = Counter(words).most_common(top_n_words)
        
        if word_counts:
            words_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])
            
            fig = px.bar(
                words_df,
                x='Count',
                y='Word',
                orientation='h',
                title=f'Top {top_n_words} Words in Titles'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Data Explorer
    with tab5:
        st.header("Raw Data Explorer")
        
        st.subheader("Search Papers")
        search_term = st.text_input("Search in titles and abstracts")
        
        if search_term:
            mask = (
                df_filtered['title'].str.contains(search_term, case=False, na=False) |
                df_filtered['abstract'].str.contains(search_term, case=False, na=False)
            )
            display_df = df_filtered[mask]
            st.info(f"Found {len(display_df):,} papers matching '{search_term}'")
        else:
            display_df = df_filtered
        
        # Column selection
        available_columns = ['title', 'abstract', 'authors', 'journal', 'publish_time', 'source_x', 'doi']
        selected_columns = st.multiselect(
            "Select columns to display",
            available_columns,
            default=['title', 'journal', 'publish_time']
        )
        
        if selected_columns:
            st.dataframe(
                display_df[selected_columns].head(100),
                use_container_width=True,
                height=400
            )
        
        # Download data
        st.subheader("Download Filtered Data")
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="covid19_filtered_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()