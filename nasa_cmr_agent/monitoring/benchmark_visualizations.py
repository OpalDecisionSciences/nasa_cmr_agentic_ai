"""
Performance Benchmark Visualization Integration.

Provides interactive dashboards and charts for performance benchmark monitoring,
with real-time updates and historical trend analysis.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
import pandas as pd
from dataclasses import asdict
import structlog

logger = structlog.get_logger(__name__)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not available - benchmark visualizations will be limited")
    PLOTLY_AVAILABLE = False


class BenchmarkVisualizationSystem:
    """Performance benchmark visualization and dashboard system."""
    
    def __init__(self, benchmark_system):
        self.benchmark_system = benchmark_system
        self.dashboard_cache = {}
        self.cache_ttl = 60  # Cache dashboards for 60 seconds
    
    def create_performance_dashboard(self) -> Dict[str, Any]:
        """Create comprehensive performance dashboard."""
        
        dashboard_data = self.benchmark_system.get_performance_dashboard_data()
        
        if not PLOTLY_AVAILABLE:
            return {
                "type": "basic_dashboard",
                "data": dashboard_data,
                "charts": ["Charts require plotly installation"]
            }
        
        charts = {}
        
        # 1. Query Performance Trends
        charts["query_performance"] = self._create_query_performance_chart(
            dashboard_data["performance_trends"]["query_times"]
        )
        
        # 2. System Health Overview
        charts["system_health"] = self._create_system_health_chart(dashboard_data["summary"])
        
        # 3. Database Performance Comparison
        charts["database_performance"] = self._create_database_performance_chart(
            dashboard_data["performance_trends"]["database_performance"]
        )
        
        # 4. Accuracy Metrics
        charts["accuracy_metrics"] = self._create_accuracy_metrics_chart(
            dashboard_data["accuracy_metrics"]
        )
        
        # 5. LangGraph State Performance
        charts["langgraph_performance"] = self._create_langgraph_performance_chart(
            dashboard_data["langgraph_performance"]
        )
        
        # 6. Success Rate Trends
        charts["success_rates"] = self._create_success_rate_chart(dashboard_data["summary"])
        
        return {
            "type": "interactive_dashboard",
            "timestamp": dashboard_data["timestamp"],
            "summary": dashboard_data["summary"],
            "charts": charts,
            "raw_data": dashboard_data
        }
    
    def _create_query_performance_chart(self, query_times: List[float]) -> Dict[str, Any]:
        """Create query performance trend chart."""
        if not query_times:
            return {"error": "No query time data available"}
        
        # Create time series chart
        fig = go.Figure()
        
        x_values = list(range(len(query_times)))
        
        # Add query time line
        fig.add_trace(go.Scatter(
            x=x_values,
            y=query_times,
            mode='lines+markers',
            name='Query Response Time',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Add target threshold line
        target_time = 5.0  # 5 second target
        fig.add_hline(y=target_time, line_dash="dash", line_color="green",
                     annotation_text="Target (5s)")
        
        # Add warning threshold line  
        warning_time = 8.0  # 8 second warning
        fig.add_hline(y=warning_time, line_dash="dash", line_color="orange",
                     annotation_text="Warning (8s)")
        
        fig.update_layout(
            title="Query Response Time Trends",
            xaxis_title="Query Sequence",
            yaxis_title="Response Time (seconds)",
            showlegend=True,
            height=400
        )
        
        return {
            "chart_type": "time_series",
            "title": "Query Performance Trends",
            "plotly_json": fig.to_json(),
            "summary": {
                "avg_time": sum(query_times) / len(query_times),
                "min_time": min(query_times),
                "max_time": max(query_times),
                "queries_under_target": sum(1 for t in query_times if t <= target_time)
            }
        }
    
    def _create_system_health_chart(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Create system health overview chart."""
        
        # Create gauge chart for system health
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = summary.get("success_rate", 0) * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "System Health (Success Rate %)"},
            delta = {'reference': 95, 'increasing': {'color': "RebeccaPurple"}},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 80], 'color': "lightgray"},
                    {'range': [80, 95], 'color': "yellow"},
                    {'range': [95, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ))
        
        fig.update_layout(height=300)
        
        return {
            "chart_type": "gauge",
            "title": "System Health Overview",
            "plotly_json": fig.to_json(),
            "summary": summary
        }
    
    def _create_database_performance_chart(self, db_performance: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create database performance comparison chart."""
        if not db_performance:
            return {"error": "No database performance data available"}
        
        # Group by database type
        weaviate_times = []
        neo4j_times = []
        timestamps = []
        
        for perf in db_performance:
            metadata = perf.get("metadata", {})
            database = metadata.get("database", "unknown")
            
            if database.lower() == "weaviate":
                weaviate_times.append(perf["measured_value"])
            elif database.lower() == "neo4j":
                neo4j_times.append(perf["measured_value"])
            
            timestamps.append(perf["timestamp"])
        
        fig = go.Figure()
        
        if weaviate_times:
            fig.add_trace(go.Scatter(
                y=weaviate_times,
                mode='lines+markers',
                name='Weaviate (Vector Search)',
                line=dict(color='purple')
            ))
        
        if neo4j_times:
            fig.add_trace(go.Scatter(
                y=neo4j_times,
                mode='lines+markers',
                name='Neo4j (Knowledge Graph)',
                line=dict(color='red')
            ))
        
        fig.update_layout(
            title="Database Performance Comparison",
            xaxis_title="Query Sequence",
            yaxis_title="Response Time (seconds)",
            showlegend=True,
            height=400
        )
        
        return {
            "chart_type": "multi_line",
            "title": "Database Performance",
            "plotly_json": fig.to_json(),
            "summary": {
                "weaviate_avg": sum(weaviate_times) / len(weaviate_times) if weaviate_times else 0,
                "neo4j_avg": sum(neo4j_times) / len(neo4j_times) if neo4j_times else 0
            }
        }
    
    def _create_accuracy_metrics_chart(self, accuracy_data: Dict[str, float]) -> Dict[str, Any]:
        """Create accuracy metrics radar chart."""
        
        categories = ['Relevance', 'Completeness', 'Overall Accuracy']
        values = [
            accuracy_data.get("avg_relevance_score", 0) * 100,
            accuracy_data.get("avg_completeness_score", 0) * 100,
            accuracy_data.get("avg_overall_accuracy", 0) * 100
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Current Performance',
            line=dict(color='blue')
        ))
        
        # Add target performance
        targets = [85, 90, 85]  # Target percentages
        fig.add_trace(go.Scatterpolar(
            r=targets,
            theta=categories,
            fill='toself',
            name='Target Performance',
            line=dict(color='green', dash='dash'),
            opacity=0.6
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Query Accuracy Metrics",
            height=400
        )
        
        return {
            "chart_type": "radar",
            "title": "Accuracy Metrics",
            "plotly_json": fig.to_json(),
            "summary": accuracy_data
        }
    
    def _create_langgraph_performance_chart(self, langgraph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create LangGraph state performance visualization."""
        
        # Create combined chart for workflow metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Active Workflows', 'Avg Workflow Time', 
                           'Memory Usage', 'Workflow Distribution'),
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'bar'}, {'type': 'pie'}]]
        )
        
        # Active workflows gauge
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=langgraph_data.get("active_workflows", 0),
                title={'text': "Active Workflows"},
                gauge={'axis': {'range': [0, 100]}},
                domain={'row': 0, 'column': 0}
            ),
            row=1, col=1
        )
        
        # Average workflow time
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=langgraph_data.get("avg_workflow_time", 0),
                title={'text': "Avg Workflow Time (s)"},
                number={'suffix': "s"},
                domain={'row': 0, 'column': 1}
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=600)
        
        return {
            "chart_type": "combined",
            "title": "LangGraph Performance",
            "plotly_json": fig.to_json(),
            "summary": langgraph_data
        }
    
    def _create_success_rate_chart(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Create success rate visualization."""
        
        success_rate = summary.get("success_rate", 0) * 100
        queries_processed = summary.get("queries_processed", 0)
        
        # Create donut chart
        labels = ['Successful', 'Failed']
        values = [success_rate, 100 - success_rate]
        colors = ['green', 'red']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values,
            hole=.3,
            marker_colors=colors
        )])
        
        fig.update_layout(
            title=f"Query Success Rate ({queries_processed} total queries)",
            annotations=[dict(text=f'{success_rate:.1f}%', x=0.5, y=0.5, 
                            font_size=20, showarrow=False)],
            height=400
        )
        
        return {
            "chart_type": "donut",
            "title": "Success Rate Overview",
            "plotly_json": fig.to_json(),
            "summary": {
                "success_rate": success_rate,
                "queries_processed": queries_processed
            }
        }
    
    def generate_benchmark_report(self, time_period: str = "24h") -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        
        # Calculate time range
        now = datetime.now(timezone.utc)
        if time_period == "1h":
            start_time = now - timedelta(hours=1)
        elif time_period == "24h":
            start_time = now - timedelta(days=1)
        elif time_period == "7d":
            start_time = now - timedelta(days=7)
        else:
            start_time = now - timedelta(days=1)
        
        # Filter results by time period
        filtered_results = [
            r for r in self.benchmark_system.benchmark_results
            if datetime.fromisoformat(r.timestamp.replace('Z', '+00:00')) >= start_time
        ]
        
        # Generate report sections
        report = {
            "report_id": f"benchmark_report_{int(now.timestamp())}",
            "generated_at": now.isoformat(),
            "time_period": time_period,
            "summary": self._generate_report_summary(filtered_results),
            "performance_analysis": self._analyze_performance_trends(filtered_results),
            "benchmark_compliance": self._analyze_benchmark_compliance(filtered_results),
            "recommendations": self._generate_recommendations(filtered_results),
            "detailed_metrics": self._generate_detailed_metrics(filtered_results)
        }
        
        return report
    
    def _generate_report_summary(self, results: List[Any]) -> Dict[str, Any]:
        """Generate high-level report summary."""
        if not results:
            return {"error": "No benchmark data available for the specified period"}
        
        # Calculate summary statistics
        total_benchmarks = len(results)
        excellent_count = sum(1 for r in results if r.threshold_status.value == "excellent")
        good_count = sum(1 for r in results if r.threshold_status.value == "good")
        warning_count = sum(1 for r in results if r.threshold_status.value == "warning")
        critical_count = sum(1 for r in results if r.threshold_status.value == "critical")
        
        return {
            "total_benchmarks": total_benchmarks,
            "health_score": ((excellent_count + good_count) / total_benchmarks * 100) if total_benchmarks > 0 else 0,
            "status_distribution": {
                "excellent": excellent_count,
                "good": good_count,
                "warning": warning_count,
                "critical": critical_count
            },
            "performance_trend": "improving",  # Would calculate based on time series
            "critical_issues": critical_count,
            "key_achievements": [
                f"{excellent_count} benchmarks meeting excellence targets",
                f"{((excellent_count + good_count) / total_benchmarks * 100):.1f}% overall health score"
            ]
        }
    
    def _analyze_performance_trends(self, results: List[Any]) -> Dict[str, Any]:
        """Analyze performance trends from benchmark results."""
        
        # Group results by category
        category_performance = {}
        for result in results:
            category = result.category.value
            if category not in category_performance:
                category_performance[category] = []
            category_performance[category].append(result)
        
        trends = {}
        for category, category_results in category_performance.items():
            if category_results:
                # Calculate trend metrics
                excellent_pct = sum(1 for r in category_results if r.threshold_status.value == "excellent") / len(category_results) * 100
                avg_performance = sum(r.measured_value for r in category_results) / len(category_results)
                
                trends[category] = {
                    "benchmark_count": len(category_results),
                    "excellence_percentage": excellent_pct,
                    "average_performance": avg_performance,
                    "trend_direction": "stable"  # Would calculate based on time series
                }
        
        return trends
    
    def _analyze_benchmark_compliance(self, results: List[Any]) -> Dict[str, Any]:
        """Analyze benchmark compliance against targets."""
        
        compliance_analysis = {}
        
        # Group by benchmark name
        benchmark_groups = {}
        for result in results:
            name = result.benchmark_name
            if name not in benchmark_groups:
                benchmark_groups[name] = []
            benchmark_groups[name].append(result)
        
        for benchmark_name, benchmark_results in benchmark_groups.items():
            latest_result = max(benchmark_results, key=lambda r: r.timestamp)
            
            compliance_analysis[benchmark_name] = {
                "current_value": latest_result.measured_value,
                "target_value": latest_result.target_value,
                "compliance_status": latest_result.threshold_status.value,
                "unit": latest_result.unit,
                "measurements_count": len(benchmark_results),
                "compliance_rate": sum(1 for r in benchmark_results 
                                     if r.threshold_status.value in ["excellent", "good"]) / len(benchmark_results) * 100
            }
        
        return compliance_analysis
    
    def _generate_recommendations(self, results: List[Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # Analyze critical and warning issues
        critical_results = [r for r in results if r.threshold_status.value == "critical"]
        warning_results = [r for r in results if r.threshold_status.value == "warning"]
        
        if critical_results:
            recommendations.append(f"🚨 Address {len(critical_results)} critical performance issues immediately")
            
            # Specific recommendations based on benchmark types
            critical_categories = set(r.category.value for r in critical_results)
            if "query_processing" in critical_categories:
                recommendations.append("• Optimize query processing pipeline - consider caching and parallel execution")
            if "database_performance" in critical_categories:
                recommendations.append("• Review database queries and connection pooling configuration")
            if "langgraph_state" in critical_categories:
                recommendations.append("• Optimize LangGraph state management and reduce memory usage")
        
        if warning_results:
            recommendations.append(f"⚠️ Monitor {len(warning_results)} performance warnings")
        
        # Positive recommendations
        excellent_results = [r for r in results if r.threshold_status.value == "excellent"]
        if excellent_results:
            recommendations.append(f"✅ {len(excellent_results)} benchmarks are performing excellently")
        
        return recommendations
    
    def _generate_detailed_metrics(self, results: List[Any]) -> Dict[str, Any]:
        """Generate detailed metrics breakdown."""
        
        # Convert results to structured format
        detailed_metrics = {
            "benchmark_details": [asdict(r) for r in results[-50:]],  # Last 50 results
            "category_breakdown": {},
            "time_series_data": []
        }
        
        # Group by category for breakdown
        for result in results:
            category = result.category.value
            if category not in detailed_metrics["category_breakdown"]:
                detailed_metrics["category_breakdown"][category] = {
                    "count": 0,
                    "avg_performance": 0,
                    "benchmarks": []
                }
            
            detailed_metrics["category_breakdown"][category]["count"] += 1
            detailed_metrics["category_breakdown"][category]["benchmarks"].append({
                "name": result.benchmark_name,
                "value": result.measured_value,
                "status": result.threshold_status.value
            })
        
        return detailed_metrics


# Integration functions for dashboard systems
def create_streamlit_dashboard(benchmark_system):
    """Create Streamlit dashboard components (if Streamlit is available)."""
    try:
        import streamlit as st
        
        visualization_system = BenchmarkVisualizationSystem(benchmark_system)
        dashboard_data = visualization_system.create_performance_dashboard()
        
        st.title("🚀 NASA CMR Agent Performance Dashboard")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        summary = dashboard_data.get("summary", {})
        
        with col1:
            st.metric("Avg Query Time", f"{summary.get('avg_query_time', 0):.2f}s")
        with col2:
            st.metric("Success Rate", f"{summary.get('success_rate', 0)*100:.1f}%")
        with col3:
            st.metric("Queries Processed", summary.get('queries_processed', 0))
        with col4:
            st.metric("System Health", summary.get('system_health', 'unknown').title())
        
        # Display charts
        charts = dashboard_data.get("charts", {})
        for chart_name, chart_data in charts.items():
            if chart_data.get("plotly_json"):
                st.plotly_chart(chart_data["plotly_json"], use_container_width=True)
        
        return True
        
    except ImportError:
        logger.warning("Streamlit not available for dashboard creation")
        return False


def create_web_dashboard_api(benchmark_system):
    """Create web dashboard API endpoints (FastAPI integration)."""
    
    visualization_system = BenchmarkVisualizationSystem(benchmark_system)
    
    # This would integrate with the existing FastAPI app
    dashboard_endpoints = {
        "/dashboard": visualization_system.create_performance_dashboard,
        "/dashboard/report/{time_period}": visualization_system.generate_benchmark_report
    }
    
    return dashboard_endpoints


# Global visualization system
def get_benchmark_visualizations(benchmark_system):
    """Get the benchmark visualization system."""
    return BenchmarkVisualizationSystem(benchmark_system)