"""
NASA CMR AI Agent - Streamlit UI

Interactive web interface for the NASA CMR AI Agent with feedback functionality.
"""

import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Configure page
st.set_page_config(
    page_title="NASA CMR AI Agent",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_response' not in st.session_state:
    st.session_state.current_response = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = {}


def submit_query(query_text):
    """Submit query to the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": query_text, "include_visualizations": True}
        )
        if response.status_code == 200:
            result = response.json()
            st.session_state.current_response = result
            st.session_state.query_history.append({
                "timestamp": datetime.now(),
                "query": query_text,
                "response": result
            })
            return result
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None


def submit_feedback(query_id, rating, helpful, issues, suggestions):
    """Submit user feedback to the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/feedback",
            json={
                "query_id": query_id,
                "rating": rating,
                "helpful": helpful,
                "issues": issues,
                "suggestions": suggestions
            }
        )
        if response.status_code == 200:
            st.session_state.feedback_submitted[query_id] = True
            return True
        return False
    except Exception as e:
        st.error(f"Failed to submit feedback: {str(e)}")
        return False


def get_query_suggestions(partial_query):
    """Get query suggestions from the API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/suggestions",
            params={"partial_query": partial_query}
        )
        if response.status_code == 200:
            return response.json().get("suggestions", [])
    except:
        pass
    return []


def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🛰️ NASA CMR AI Agent")
        st.markdown("Natural language interface for Earth science data discovery")
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.header("System Information")
        
        # System status
        try:
            health_response = requests.get(f"{API_BASE_URL}/health")
            if health_response.status_code == 200:
                st.success("✅ System Online")
            else:
                st.error("❌ System Offline")
        except:
            st.error("❌ Cannot connect to API")
        
        # System capabilities
        st.subheader("Capabilities")
        try:
            capabilities = requests.get(f"{API_BASE_URL}/capabilities").json()
            with st.expander("Supported Features"):
                st.json(capabilities)
        except:
            st.info("Unable to load capabilities")
        
        # Learning Summary
        st.subheader("Adaptive Learning")
        try:
            learning_summary = requests.get(f"{API_BASE_URL}/learning/summary").json()
            if learning_summary.get("total_patterns", 0) > 0:
                st.metric("Learned Patterns", learning_summary["total_patterns"])
                st.metric("Avg Success Rate", f"{learning_summary.get('average_success_rate', 0):.2%}")
                st.metric("Total Feedback", learning_summary.get("total_feedback", 0))
        except:
            st.info("No learning data available")
        
        # Query History
        st.subheader("Query History")
        for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(f"Query {len(st.session_state.query_history) - i}"):
                st.text(item["query"][:50] + "...")
                st.caption(item["timestamp"].strftime("%H:%M:%S"))
    
    # Main content area
    tabs = st.tabs(["🔍 Query", "📊 Results", "💡 Feedback", "📈 Analytics"])
    
    # Query Tab
    with tabs[0]:
        st.header("Submit Query")
        
        # Example queries
        st.subheader("Example Queries")
        example_queries = [
            "Find precipitation datasets for drought monitoring in Sub-Saharan Africa 2015-2023",
            "Compare MODIS and VIIRS for vegetation monitoring",
            "What datasets are best for studying urban heat islands?",
            "Show me ocean temperature data near coral reefs",
            "Find atmospheric CO2 measurements from ground stations"
        ]
        
        col1, col2, col3 = st.columns(3)
        for i, example in enumerate(example_queries):
            with [col1, col2, col3][i % 3]:
                if st.button(example[:30] + "...", key=f"ex_{i}", use_container_width=True):
                    st.session_state.query_input = example
        
        # Query input
        query_text = st.text_area(
            "Enter your query:",
            height=100,
            placeholder="Describe what Earth science data you're looking for...",
            key="query_input"
        )
        
        # Query suggestions (if typing)
        if query_text and len(query_text) > 5:
            suggestions = get_query_suggestions(query_text)
            if suggestions:
                st.info("💡 Suggestions based on similar queries:")
                for suggestion in suggestions[:3]:
                    st.caption(f"• {suggestion}")
        
        # Submit button
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("🚀 Submit Query", type="primary", use_container_width=True):
                if query_text:
                    with st.spinner("Processing query..."):
                        result = submit_query(query_text)
                        if result:
                            st.success("Query processed successfully!")
                            st.rerun()
                else:
                    st.warning("Please enter a query")
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.query_input = ""
                st.rerun()
    
    # Results Tab
    with tabs[1]:
        st.header("Query Results")
        
        if st.session_state.current_response:
            response = st.session_state.current_response
            
            # Summary
            st.subheader("Summary")
            st.info(response.get("summary", "No summary available"))
            
            # Intent and execution info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Intent", response.get("intent", "Unknown"))
            with col2:
                st.metric("Success", "✅" if response.get("success") else "❌")
            with col3:
                st.metric("Execution Time", f"{response.get('total_execution_time_ms', 0)}ms")
            
            # Recommendations
            if response.get("recommendations"):
                st.subheader("Dataset Recommendations")
                for i, rec in enumerate(response["recommendations"][:5], 1):
                    with st.expander(f"{i}. {rec.get('collection', {}).get('title', 'Unknown')}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Dataset ID:** {rec.get('collection', {}).get('concept_id', 'N/A')}")
                            st.write(f"**Data Center:** {rec.get('collection', {}).get('data_center', 'N/A')}")
                            st.write(f"**Platforms:** {', '.join(rec.get('collection', {}).get('platforms', []))}")
                        with col2:
                            st.metric("Relevance Score", f"{rec.get('relevance_score', 0):.2f}")
                            st.metric("Coverage Score", f"{rec.get('coverage_score', 0):.2f}")
                            st.metric("Quality Score", f"{rec.get('quality_score', 0):.2f}")
                        
                        st.write(f"**Reasoning:** {rec.get('reasoning', 'N/A')}")
                        
                        if rec.get('granule_count'):
                            st.info(f"📦 {rec['granule_count']} granules available")
            
            # Analysis Results
            if response.get("analysis_results"):
                st.subheader("Analysis Results")
                for analysis in response["analysis_results"]:
                    with st.expander(analysis.get("analysis_type", "Analysis")):
                        st.write(f"**Methodology:** {analysis.get('methodology', 'N/A')}")
                        st.write(f"**Confidence:** {analysis.get('confidence_level', 0):.2%}")
                        
                        if analysis.get("statistics"):
                            st.write("**Key Statistics:**")
                            stats_df = pd.DataFrame([analysis["statistics"]])
                            st.dataframe(stats_df)
            
            # Warnings
            if response.get("warnings"):
                st.subheader("⚠️ Warnings")
                for warning in response["warnings"]:
                    st.warning(warning)
            
            # Follow-up suggestions
            if response.get("follow_up_suggestions"):
                st.subheader("💡 Follow-up Suggestions")
                for suggestion in response["follow_up_suggestions"]:
                    st.info(suggestion)
            
            # Quality Score (if available)
            if response.get("metadata", {}).get("quality_score"):
                st.subheader("Quality Assessment")
                quality = response["metadata"]["quality_score"]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{quality.get('accuracy', 0):.2f}")
                with col2:
                    st.metric("Completeness", f"{quality.get('completeness', 0):.2f}")
                with col3:
                    st.metric("Relevance", f"{quality.get('relevance', 0):.2f}")
                with col4:
                    st.metric("Clarity", f"{quality.get('clarity', 0):.2f}")
        else:
            st.info("No results to display. Submit a query to see results.")
    
    # Feedback Tab
    with tabs[2]:
        st.header("Provide Feedback")
        
        if st.session_state.current_response:
            query_id = st.session_state.current_response.get("query_id", "unknown")
            
            if query_id not in st.session_state.feedback_submitted:
                st.subheader("How was this response?")
                
                # Rating
                rating = st.slider("Rate the response quality:", 1, 5, 3)
                
                # Helpful
                helpful = st.checkbox("Was this response helpful?")
                
                # Issues
                st.write("Select any issues (if applicable):")
                issues = []
                col1, col2 = st.columns(2)
                with col1:
                    if st.checkbox("Inaccurate information"):
                        issues.append("inaccurate")
                    if st.checkbox("Missing datasets"):
                        issues.append("missing_datasets")
                    if st.checkbox("Wrong interpretation"):
                        issues.append("wrong_interpretation")
                with col2:
                    if st.checkbox("Too slow"):
                        issues.append("slow")
                    if st.checkbox("Unclear response"):
                        issues.append("unclear")
                    if st.checkbox("Technical errors"):
                        issues.append("errors")
                
                # Suggestions
                suggestions = st.text_area(
                    "Additional suggestions or comments:",
                    placeholder="How can we improve?"
                )
                
                # Submit feedback
                if st.button("Submit Feedback", type="primary"):
                    if submit_feedback(query_id, rating, helpful, issues, suggestions):
                        st.success("Thank you for your feedback! It helps us improve.")
                        st.balloons()
                    else:
                        st.error("Failed to submit feedback. Please try again.")
            else:
                st.success("✅ Feedback already submitted for this query. Thank you!")
        else:
            st.info("Submit a query first to provide feedback.")
    
    # Analytics Tab
    with tabs[3]:
        st.header("System Analytics")
        
        # Performance Metrics
        try:
            metrics = requests.get(f"{API_BASE_URL}/metrics").json()
            
            st.subheader("Performance Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Queries", metrics.get("total_queries", 0))
            with col2:
                st.metric("Success Rate", f"{metrics.get('success_rate', 0):.1%}")
            with col3:
                st.metric("Avg Response Time", f"{metrics.get('avg_response_time_ms', 0):.0f}ms")
            
            # Performance over time chart (if data available)
            if st.session_state.query_history:
                st.subheader("Query Performance Over Time")
                
                # Extract data for visualization
                times = [item["timestamp"] for item in st.session_state.query_history]
                exec_times = [
                    item["response"].get("total_execution_time_ms", 0) 
                    for item in st.session_state.query_history
                ]
                
                # Create line chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=times,
                    y=exec_times,
                    mode='lines+markers',
                    name='Execution Time',
                    line=dict(color='blue', width=2)
                ))
                fig.update_layout(
                    title="Query Execution Time",
                    xaxis_title="Time",
                    yaxis_title="Execution Time (ms)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Unable to load analytics: {str(e)}")


if __name__ == "__main__":
    main()