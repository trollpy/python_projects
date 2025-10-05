"""
Visualization utilities for COVID-19 research data
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Optional, Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


class Visualizer:
    """Create visualizations for COVID-19 research data"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style"""
        style = self.config.get('style', 'whitegrid')
        sns.set_style(style)
        
        plt.rcParams['figure.figsize'] = (
            self.config.get('figure_width', 12),
            self.config.get('figure_height', 6)
        )
        plt.rcParams['figure.dpi'] = self.config.get('dpi', 100)
    
    def plot_publications_by_year(
        self, 
        year_counts: pd.Series,
        title: str = "Publications by Year"
    ) -> go.Figure:
        """Create interactive bar plot of publications by year"""
        fig = px.bar(
            x=year_counts.index,
            y=year_counts.values,
            labels={'x': 'Year', 'y': 'Number of Publications'},
            title=title
        )
        fig.update_traces(
            marker_color='steelblue',
            marker_line_color='darkblue',
            marker_line_width=1.5,
            hovertemplate='<b>Year:</b> %{x}<br><b>Publications:</b> %{y:,}<extra></extra>'
        )
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Number of Publications",
            hovermode='x unified',
            plot_bgcolor='white',
            font=dict(size=12),
            title_font_size=16,
            showlegend=False
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def plot_monthly_trends(
        self,
        monthly_counts: pd.Series,
        title: str = "Monthly Publication Trends"
    ) -> go.Figure:
        """Create line plot of monthly trends"""
        monthly_df = pd.DataFrame({
            'Month': monthly_counts.index.astype(str),
            'Publications': monthly_counts.values
        })
        
        fig = px.line(
            monthly_df,
            x='Month',
            y='Publications',
            title=title,
            markers=True
        )
        
        fig.update_traces(
            line_color='#2E86AB',
            line_width=2.5,
            marker=dict(size=6, color='#A23B72'),
            hovertemplate='<b>Month:</b> %{x}<br><b>Publications:</b> %{y:,}<extra></extra>'
        )
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Number of Publications",
            hovermode='x unified',
            plot_bgcolor='white',
            font=dict(size=12),
            title_font_size=16
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='lightgray',
            tickangle=-45
        )
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def plot_top_journals(
        self,
        journal_counts: pd.Series,
        title: str = "Top Journals"
    ) -> go.Figure:
        """Create horizontal bar plot of top journals"""
        fig = px.bar(
            x=journal_counts.values,
            y=journal_counts.index,
            orientation='h',
            labels={'x': 'Number of Publications', 'y': 'Journal'},
            title=title
        )
        
        fig.update_traces(
            marker_color='coral',
            marker_line_color='darkred',
            marker_line_width=1,
            hovertemplate='<b>%{y}</b><br>Publications: %{x:,}<extra></extra>'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Number of Publications",
            yaxis_title="Journal",
            plot_bgcolor='white',
            font=dict(size=11),
            title_font_size=16,
            height=max(400, len(journal_counts) * 25)  # Dynamic height based on number of journals
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=False)
        
        return fig
    
    def plot_source_distribution(
        self,
        source_counts: pd.Series,
        title: str = "Distribution by Source"
    ) -> go.Figure:
        """Create pie chart of source distribution"""
        fig = px.pie(
            values=source_counts.values,
            names=source_counts.index,
            title=title
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Papers: %{value:,}<br>Share: %{percent}<extra></extra>',
            marker=dict(line=dict(color='white', width=2))
        )
        
        fig.update_layout(
            font=dict(size=12),
            title_font_size=16,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            )
        )
        
        return fig
    
    def create_wordcloud(
        self,
        text_series: pd.Series,
        title: str = "Word Cloud"
    ) -> plt.Figure:
        """Create word cloud from text"""
        all_text = ' '.join(text_series.dropna().astype(str).tolist())
        
        if not all_text.strip():
            logger.warning("No text available for word cloud generation")
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.text(0.5, 0.5, 'No text data available', 
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=20)
            ax.axis('off')
            return fig
        
        wordcloud_config = self.config.get('wordcloud', {})
        
        wordcloud = WordCloud(
            width=wordcloud_config.get('width', 1200),
            height=wordcloud_config.get('height', 600),
            background_color='white',
            colormap=self.config.get('color_palette', 'viridis'),
            max_words=wordcloud_config.get('max_words', 150),
            relative_scaling=0.5,
            min_font_size=10,
            prefer_horizontal=0.7,
            contour_width=2,
            contour_color='steelblue'
        ).generate(all_text)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig
    
    def plot_common_words(
        self,
        word_counts: List[tuple],
        title: str = "Most Common Words"
    ) -> go.Figure:
        """Create bar plot of common words"""
        if not word_counts:
            logger.warning("No word counts available for plotting")
            return go.Figure()
        
        words_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])
        
        fig = px.bar(
            words_df,
            x='Count',
            y='Word',
            orientation='h',
            title=title
        )
        
        fig.update_traces(
            marker_color='#5E60CE',
            marker_line_color='#4C1D95',
            marker_line_width=1,
            hovertemplate='<b>%{y}</b><br>Frequency: %{x:,}<extra></extra>'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Frequency",
            yaxis_title="Word",
            plot_bgcolor='white',
            font=dict(size=11),
            title_font_size=16,
            height=max(400, len(word_counts) * 20)
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=False)
        
        return fig
    
    def plot_abstract_length_distribution(
        self,
        df: pd.DataFrame,
        title: str = "Abstract Length Distribution"
    ) -> go.Figure:
        """Create histogram of abstract lengths"""
        if 'abstract_word_count' not in df.columns:
            logger.warning("No 'abstract_word_count' column found")
            return go.Figure()
        
        # Filter out zeros and extreme outliers
        data = df[df['abstract_word_count'] > 0]['abstract_word_count']
        
        if len(data) == 0:
            logger.warning("No valid abstract length data available")
            return go.Figure()
        
        # Calculate statistics
        mean_length = data.mean()
        median_length = data.median()
        
        fig = px.histogram(
            data,
            nbins=50,
            title=title,
            labels={'value': 'Number of Words', 'count': 'Frequency'}
        )
        
        fig.update_traces(
            marker_color='#06B6D4',
            marker_line_color='#0E7490',
            marker_line_width=1,
            hovertemplate='Words: %{x}<br>Count: %{y}<extra></extra>'
        )
        
        # Add mean and median lines
        fig.add_vline(
            x=mean_length, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Mean: {mean_length:.0f}",
            annotation_position="top"
        )
        
        fig.add_vline(
            x=median_length, 
            line_dash="dot", 
            line_color="green",
            annotation_text=f"Median: {median_length:.0f}",
            annotation_position="top"
        )
        
        fig.update_layout(
            xaxis_title="Number of Words",
            yaxis_title="Frequency",
            plot_bgcolor='white',
            font=dict(size=12),
            title_font_size=16,
            showlegend=False
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def plot_collaboration_distribution(
        self,
        df: pd.DataFrame,
        title: str = "Author Collaboration Distribution"
    ) -> go.Figure:
        """Create histogram of author counts"""
        if 'author_count' not in df.columns:
            logger.warning("No 'author_count' column found")
            return go.Figure()
        
        # Filter valid data
        data = df[df['author_count'] > 0]['author_count']
        
        if len(data) == 0:
            logger.warning("No valid author count data available")
            return go.Figure()
        
        # Cap at 20 for better visualization
        data_capped = data.clip(upper=20)
        
        # Calculate statistics
        mean_authors = data.mean()
        median_authors = data.median()
        
        fig = px.histogram(
            data_capped,
            nbins=20,
            title=title,
            labels={'value': 'Number of Authors', 'count': 'Number of Papers'}
        )
        
        fig.update_traces(
            marker_color='#F59E0B',
            marker_line_color='#B45309',
            marker_line_width=1,
            hovertemplate='Authors: %{x}<br>Papers: %{y}<extra></extra>'
        )
        
        # Add mean and median lines
        fig.add_vline(
            x=mean_authors,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_authors:.1f}",
            annotation_position="top"
        )
        
        fig.add_vline(
            x=median_authors,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Median: {median_authors:.0f}",
            annotation_position="top"
        )
        
        fig.update_layout(
            xaxis_title="Number of Authors",
            yaxis_title="Number of Papers",
            plot_bgcolor='white',
            font=dict(size=12),
            title_font_size=16,
            showlegend=False
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def plot_heatmap(
        self,
        data: pd.DataFrame,
        title: str = "Heatmap",
        x_label: str = "X",
        y_label: str = "Y"
    ) -> go.Figure:
        """Create heatmap visualization"""
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale='Viridis',
            hovertemplate='%{y}, %{x}<br>Value: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            font=dict(size=12),
            title_font_size=16
        )
        
        return fig
    
    def plot_time_series(
        self,
        dates: pd.Series,
        values: pd.Series,
        title: str = "Time Series",
        y_label: str = "Value"
    ) -> go.Figure:
        """Create time series plot"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            name='Data',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=4, color='#A23B72'),
            hovertemplate='Date: %{x}<br>Value: %{y:,}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=y_label,
            plot_bgcolor='white',
            font=dict(size=12),
            title_font_size=16,
            hovermode='x unified'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def plot_stacked_bar(
        self,
        data: pd.DataFrame,
        title: str = "Stacked Bar Chart"
    ) -> go.Figure:
        """Create stacked bar chart"""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2
        
        for i, col in enumerate(data.columns):
            fig.add_trace(go.Bar(
                name=col,
                x=data.index,
                y=data[col],
                marker_color=colors[i % len(colors)],
                hovertemplate='%{x}<br>%{y:,}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            barmode='stack',
            plot_bgcolor='white',
            font=dict(size=12),
            title_font_size=16,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def plot_scatter(
        self,
        x: pd.Series,
        y: pd.Series,
        title: str = "Scatter Plot",
        x_label: str = "X",
        y_label: str = "Y",
        color: Optional[pd.Series] = None,
        size: Optional[pd.Series] = None
    ) -> go.Figure:
        """Create scatter plot"""
        fig = px.scatter(
            x=x,
            y=y,
            title=title,
            labels={'x': x_label, 'y': y_label},
            color=color,
            size=size
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            font=dict(size=12),
            title_font_size=16
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def plot_box(
        self,
        data: pd.DataFrame,
        title: str = "Box Plot",
        y_label: str = "Value"
    ) -> go.Figure:
        """Create box plot"""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Pastel
        
        for i, col in enumerate(data.columns):
            fig.add_trace(go.Box(
                y=data[col],
                name=col,
                marker_color=colors[i % len(colors)],
                boxmean='sd'
            ))
        
        fig.update_layout(
            title=title,
            yaxis_title=y_label,
            plot_bgcolor='white',
            font=dict(size=12),
            title_font_size=16,
            showlegend=True
        )
        
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        title: str = "Correlation Matrix"
    ) -> go.Figure:
        """Create correlation matrix heatmap"""
        corr_matrix = data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            font=dict(size=12),
            title_font_size=16,
            width=700,
            height=700
        )
        
        return fig
    
    def save_figure(
        self,
        fig,
        filename: str,
        format: str = 'png',
        **kwargs
    ) -> None:
        """Save figure to file"""
        try:
            if isinstance(fig, go.Figure):
                # Plotly figure
                if format == 'html':
                    fig.write_html(filename, **kwargs)
                else:
                    fig.write_image(filename, **kwargs)
            else:
                # Matplotlib figure
                fig.savefig(filename, format=format, bbox_inches='tight', **kwargs)
            
            logger.info(f"Figure saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving figure to {filename}: {e}")
            raise