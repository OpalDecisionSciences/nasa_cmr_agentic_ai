import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import base64
from io import BytesIO

from ..models.schemas import DatasetRecommendation, AnalysisResult, CMRCollection


class VisualizationService:
    """
    Advanced visualization service for NASA CMR data analysis results.
    
    Provides interactive and static visualizations for:
    - Dataset recommendations and scores
    - Temporal coverage analysis
    - Spatial coverage maps
    - Gap analysis charts
    - Performance metrics dashboards
    """
    
    def __init__(self):
        # Set up plotting styles
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Plotly template
        self.plotly_template = "plotly_white"
    
    def create_recommendation_dashboard(
        self, 
        recommendations: List[DatasetRecommendation]
    ) -> Dict[str, Any]:
        """Create comprehensive dashboard for dataset recommendations."""
        
        if not recommendations:
            return {"error": "No recommendations to visualize"}
        
        visualizations = {}
        
        # 1. Scoring comparison chart
        visualizations["scoring_comparison"] = self._create_scoring_comparison(recommendations)
        
        # 2. Platform distribution
        visualizations["platform_distribution"] = self._create_platform_distribution(recommendations)
        
        # 3. Temporal coverage timeline
        visualizations["temporal_coverage"] = self._create_temporal_coverage_chart(recommendations)
        
        # 4. Quality vs Accessibility scatter plot
        visualizations["quality_accessibility"] = self._create_quality_accessibility_plot(recommendations)
        
        # 5. Granule count distribution
        visualizations["granule_distribution"] = self._create_granule_distribution(recommendations)
        
        return visualizations
    
    def create_coverage_analysis_charts(
        self, 
        analysis_results: List[AnalysisResult]
    ) -> Dict[str, Any]:
        """Create charts for temporal and spatial coverage analysis."""
        
        visualizations = {}
        
        for result in analysis_results:
            if result.analysis_type == "coverage_analysis":
                # Temporal coverage charts
                if "temporal" in result.results:
                    temporal_data = result.results["temporal"]
                    visualizations["temporal_analysis"] = self._create_temporal_analysis_chart(temporal_data)
                
                # Spatial coverage maps
                if "spatial" in result.results:
                    spatial_data = result.results["spatial"]
                    visualizations["spatial_coverage"] = self._create_spatial_coverage_map(spatial_data)
            
            elif result.analysis_type == "temporal_gap_analysis":
                visualizations["gap_analysis"] = self._create_gap_analysis_chart(result.results)
        
        return visualizations
    
    def create_spatial_coverage_map(
        self, 
        recommendations: List[DatasetRecommendation],
        query_spatial_constraint: Optional[Dict[str, float]] = None
    ) -> str:
        """Create interactive map showing spatial coverage of datasets."""
        
        # Create base map
        if query_spatial_constraint:
            center_lat = (query_spatial_constraint.get('north', 0) + 
                         query_spatial_constraint.get('south', 0)) / 2
            center_lon = (query_spatial_constraint.get('east', 0) + 
                         query_spatial_constraint.get('west', 0)) / 2
        else:
            center_lat, center_lon = 0, 0
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=2,
            tiles='OpenStreetMap'
        )
        
        # Add query region if provided
        if query_spatial_constraint and all(
            k in query_spatial_constraint for k in ['north', 'south', 'east', 'west']
        ):
            query_bounds = [
                [query_spatial_constraint['south'], query_spatial_constraint['west']],
                [query_spatial_constraint['north'], query_spatial_constraint['east']]
            ]
            
            folium.Rectangle(
                bounds=query_bounds,
                color='red',
                fill=False,
                weight=3,
                popup='Query Region'
            ).add_to(m)
        
        # Add dataset coverage areas
        colors = ['blue', 'green', 'purple', 'orange', 'darkred']
        
        for i, rec in enumerate(recommendations[:5]):  # Limit to top 5
            if rec.collection.spatial_coverage:
                spatial = rec.collection.spatial_coverage
                if all(k in spatial for k in ['north', 'south', 'east', 'west']):
                    bounds = [
                        [spatial['south'], spatial['west']],
                        [spatial['north'], spatial['east']]
                    ]
                    
                    color = colors[i % len(colors)]
                    
                    folium.Rectangle(
                        bounds=bounds,
                        color=color,
                        fillColor=color,
                        fillOpacity=0.2,
                        weight=2,
                        popup=f"{rec.collection.title}<br>Score: {rec.relevance_score:.2f}"
                    ).add_to(m)
        
        # Convert map to HTML string
        return m._repr_html_()
    
    def _create_scoring_comparison(self, recommendations: List[DatasetRecommendation]) -> str:
        """Create scoring comparison chart."""
        
        # Prepare data
        data = []
        for rec in recommendations[:10]:  # Top 10
            data.append({
                'Dataset': rec.collection.title[:30] + '...' if len(rec.collection.title) > 30 else rec.collection.title,
                'Relevance': rec.relevance_score,
                'Coverage': rec.coverage_score, 
                'Quality': rec.quality_score,
                'Accessibility': rec.accessibility_score
            })
        
        df = pd.DataFrame(data)
        
        # Create plotly chart
        fig = px.bar(
            df.melt(id_vars=['Dataset'], var_name='Metric', value_name='Score'),
            x='Dataset',
            y='Score',
            color='Metric',
            title='Dataset Recommendation Scores',
            template=self.plotly_template
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            showlegend=True
        )
        
        return fig.to_html(include_plotlyjs='inline')
    
    def _create_platform_distribution(self, recommendations: List[DatasetRecommendation]) -> str:
        """Create platform distribution pie chart."""
        
        platform_counts = {}
        for rec in recommendations:
            for platform in rec.collection.platforms:
                platform_counts[platform] = platform_counts.get(platform, 0) + 1
        
        if not platform_counts:
            return "<p>No platform data available</p>"
        
        fig = px.pie(
            values=list(platform_counts.values()),
            names=list(platform_counts.keys()),
            title='Dataset Distribution by Platform',
            template=self.plotly_template
        )
        
        return fig.to_html(include_plotlyjs='inline')
    
    def _create_temporal_coverage_chart(self, recommendations: List[DatasetRecommendation]) -> str:
        """Create temporal coverage timeline chart."""
        
        timeline_data = []
        
        for rec in recommendations:
            if (rec.collection.temporal_coverage and 
                rec.collection.temporal_coverage.get('start') and
                rec.collection.temporal_coverage.get('end')):
                
                try:
                    start_date = datetime.fromisoformat(
                        rec.collection.temporal_coverage['start'].replace('Z', '+00:00')
                    )
                    end_date = datetime.fromisoformat(
                        rec.collection.temporal_coverage['end'].replace('Z', '+00:00')
                    )
                    
                    timeline_data.append({
                        'Dataset': rec.collection.title[:25] + '...' if len(rec.collection.title) > 25 else rec.collection.title,
                        'Start': start_date,
                        'End': end_date,
                        'Score': rec.relevance_score
                    })
                    
                except (ValueError, TypeError):
                    continue
        
        if not timeline_data:
            return "<p>No temporal coverage data available</p>"
        
        df = pd.DataFrame(timeline_data)
        
        fig = px.timeline(
            df,
            x_start='Start',
            x_end='End', 
            y='Dataset',
            color='Score',
            title='Dataset Temporal Coverage',
            template=self.plotly_template
        )
        
        fig.update_layout(height=max(300, len(timeline_data) * 30))
        
        return fig.to_html(include_plotlyjs='inline')
    
    def _create_quality_accessibility_plot(self, recommendations: List[DatasetRecommendation]) -> str:
        """Create quality vs accessibility scatter plot."""
        
        data = []
        for rec in recommendations:
            data.append({
                'Quality': rec.quality_score,
                'Accessibility': rec.accessibility_score,
                'Dataset': rec.collection.title,
                'Relevance': rec.relevance_score,
                'Coverage': rec.coverage_score
            })
        
        df = pd.DataFrame(data)
        
        fig = px.scatter(
            df,
            x='Quality',
            y='Accessibility',
            size='Relevance',
            color='Coverage',
            hover_data=['Dataset'],
            title='Dataset Quality vs Accessibility',
            template=self.plotly_template
        )
        
        return fig.to_html(include_plotlyjs='inline')
    
    def _create_granule_distribution(self, recommendations: List[DatasetRecommendation]) -> str:
        """Create granule count distribution chart."""
        
        granule_data = []
        for rec in recommendations:
            if rec.granule_count is not None:
                granule_data.append({
                    'Dataset': rec.collection.title[:20] + '...' if len(rec.collection.title) > 20 else rec.collection.title,
                    'Granule_Count': rec.granule_count,
                    'Score': rec.relevance_score
                })
        
        if not granule_data:
            return "<p>No granule count data available</p>"
        
        df = pd.DataFrame(granule_data)
        
        fig = px.bar(
            df,
            x='Dataset',
            y='Granule_Count',
            color='Score',
            title='Available Granules by Dataset',
            template=self.plotly_template
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        
        return fig.to_html(include_plotlyjs='inline')
    
    def _create_temporal_analysis_chart(self, temporal_data: Dict[str, Any]) -> str:
        """Create temporal analysis chart."""
        
        # Create summary chart of temporal coverage
        stats = []
        
        if 'total_temporal_span_days' in temporal_data:
            stats.append({
                'Metric': 'Total Span (Years)',
                'Value': temporal_data['total_temporal_span_days'] / 365.25
            })
        
        if 'collection_count_with_temporal' in temporal_data:
            stats.append({
                'Metric': 'Collections with Temporal Data',
                'Value': temporal_data['collection_count_with_temporal']
            })
        
        if 'granule_count' in temporal_data:
            stats.append({
                'Metric': 'Total Granules',
                'Value': temporal_data['granule_count']
            })
        
        if not stats:
            return "<p>No temporal statistics available</p>"
        
        df = pd.DataFrame(stats)
        
        fig = px.bar(
            df,
            x='Metric',
            y='Value',
            title='Temporal Coverage Statistics',
            template=self.plotly_template
        )
        
        return fig.to_html(include_plotlyjs='inline')
    
    def _create_spatial_coverage_map(self, spatial_data: Dict[str, Any]) -> str:
        """Create spatial coverage visualization."""
        
        if 'bounding_box' not in spatial_data:
            return "<p>No spatial coverage data available</p>"
        
        bbox = spatial_data['bounding_box']
        
        # Create simple map showing overall coverage
        center_lat = (bbox['north'] + bbox['south']) / 2
        center_lon = (bbox['east'] + bbox['west']) / 2
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=2)
        
        # Add bounding box
        bounds = [
            [bbox['south'], bbox['west']],
            [bbox['north'], bbox['east']]
        ]
        
        folium.Rectangle(
            bounds=bounds,
            color='blue',
            fill=True,
            fillOpacity=0.3,
            popup=f"Coverage Area<br>Collections: {spatial_data.get('collections_with_spatial', 0)}"
        ).add_to(m)
        
        return m._repr_html_()
    
    def _create_gap_analysis_chart(self, gap_data: Dict[str, Any]) -> str:
        """Create temporal gap analysis chart."""
        
        if 'statistics' not in gap_data:
            return "<p>No gap analysis data available</p>"
        
        stats = gap_data['statistics']
        
        metrics = []
        if 'total_gaps' in stats:
            metrics.append({'Metric': 'Total Gaps', 'Value': stats['total_gaps']})
        if 'average_gap_days' in stats:
            metrics.append({'Metric': 'Average Gap (Days)', 'Value': stats['average_gap_days']})
        if 'longest_gap_days' in stats:
            metrics.append({'Metric': 'Longest Gap (Days)', 'Value': stats['longest_gap_days']})
        
        if not metrics:
            return "<p>No gap statistics available</p>"
        
        df = pd.DataFrame(metrics)
        
        fig = px.bar(
            df,
            x='Metric',
            y='Value',
            title='Temporal Gap Analysis',
            template=self.plotly_template,
            color='Value',
            color_continuous_scale='Reds'
        )
        
        return fig.to_html(include_plotlyjs='inline')
    
    def create_performance_dashboard(self, metrics_data: Dict[str, Any]) -> str:
        """Create system performance dashboard."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Query Volume', 'Processing Times', 'Success Rate', 'Error Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Query volume trend (example data)
        if 'query_volume' in metrics_data:
            volume_data = metrics_data['query_volume']
            timestamps = [point['timestamp'] for point in volume_data]
            counts = [point['count'] for point in volume_data]
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=counts, name='Query Volume'),
                row=1, col=1
            )
        
        # Processing times
        if 'processing_times' in metrics_data:
            times = metrics_data['processing_times']
            fig.add_trace(
                go.Histogram(x=times, name='Processing Time Distribution'),
                row=1, col=2
            )
        
        # Success rate indicator
        queries_info = metrics_data.get('queries', {})
        success_rate = queries_info.get('success_rate', 0) * 100
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=success_rate,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Success Rate (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkgreen"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "lightgreen"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 90}}
            ),
            row=2, col=1
        )
        
        # Error rate indicator
        error_info = metrics_data.get('errors', {})
        error_rate = error_info.get('error_rate', 0) * 100
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=error_rate,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Error Rate (%)"},
                gauge={'axis': {'range': [0, 20]},
                       'bar': {'color': "darkred"},
                       'steps': [{'range': [0, 5], 'color': "lightgreen"},
                                {'range': [5, 10], 'color': "yellow"},
                                {'range': [10, 20], 'color': "lightcoral"}]}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="NASA CMR Agent Performance Dashboard",
            template=self.plotly_template
        )
        
        return fig.to_html(include_plotlyjs='inline')
    
    def export_chart_as_image(self, chart_html: str, format: str = 'png') -> bytes:
        """Export chart as image (requires additional dependencies)."""
        # This would require selenium or kaleido for plotly
        # For now, return placeholder
        return b"Chart export not implemented"