#!/usr/bin/env python3
"""
NASA CMR AI Agent - Basic Usage Examples

This script demonstrates how to use the NASA CMR AI Agent system
to perform natural language queries for Earth science data discovery.
"""

import asyncio
import os
from nasa_cmr_agent.core.graph import CMRAgentGraph
from nasa_cmr_agent.models import SystemResponse


async def example_precipitation_query():
    """Example: Finding precipitation datasets for drought monitoring."""
    
    print("=== Precipitation Data Discovery Example ===")
    
    # Initialize the agent system
    agent = CMRAgentGraph()
    
    # Example query from the technical requirements
    query = ("Compare precipitation datasets suitable for drought monitoring "
             "in Sub-Saharan Africa between 2015-2023, considering both "
             "satellite and ground-based observations, and identify gaps "
             "in temporal coverage.")
    
    print(f"Query: {query}")
    print("\nProcessing...")
    
    try:
        # Process the query
        response = await agent.process_query(query)
        
        # Display results
        print(f"\nQuery ID: {response.query_id}")
        print(f"Success: {response.success}")
        print(f"Intent: {response.intent}")
        print(f"Processing Time: {response.total_execution_time_ms}ms")
        
        print(f"\n=== Summary ===")
        print(response.summary)
        
        print(f"\n=== Recommendations ({len(response.recommendations)}) ===")
        for i, rec in enumerate(response.recommendations[:5], 1):
            print(f"\n{i}. {rec.collection.title}")
            print(f"   Relevance: {rec.relevance_score:.2f}")
            print(f"   Coverage: {rec.coverage_score:.2f}")
            print(f"   Quality: {rec.quality_score:.2f}")
            print(f"   Reasoning: {rec.reasoning}")
            if rec.temporal_gaps:
                print(f"   Temporal Gaps: {len(rec.temporal_gaps)} identified")
        
        print(f"\n=== Analysis Results ({len(response.analysis_results)}) ===")
        for result in response.analysis_results:
            print(f"- {result.analysis_type}: {result.methodology}")
            if result.statistics:
                for key, value in result.statistics.items():
                    print(f"  {key}: {value}")
        
        if response.warnings:
            print(f"\n=== Warnings ===")
            for warning in response.warnings:
                print(f"- {warning}")
        
        if response.follow_up_suggestions:
            print(f"\n=== Follow-up Suggestions ===")
            for suggestion in response.follow_up_suggestions:
                print(f"- {suggestion}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Clean up connections
        await agent.cmr_api_agent.close()


async def example_urban_heat_query():
    """Example: Multi-disciplinary dataset discovery."""
    
    print("\n\n=== Urban Heat Island and Air Quality Example ===")
    
    agent = CMRAgentGraph()
    
    # Example query focusing on urban research
    query = ("What datasets would be best for studying the relationship "
             "between urban heat islands and air quality in megacities?")
    
    print(f"Query: {query}")
    print("\nProcessing...")
    
    try:
        response = await agent.process_query(query)
        
        print(f"\nQuery ID: {response.query_id}")
        print(f"Success: {response.success}")
        print(f"Intent: {response.intent}")
        
        print(f"\n=== Summary ===")
        print(response.summary)
        
        print(f"\n=== Top Recommendations ===")
        for i, rec in enumerate(response.recommendations[:3], 1):
            print(f"\n{i}. {rec.collection.title}")
            print(f"   Platform: {', '.join(rec.collection.platforms) if rec.collection.platforms else 'N/A'}")
            print(f"   Data Center: {rec.collection.data_center}")
            print(f"   Overall Score: {(rec.relevance_score + rec.coverage_score + rec.quality_score) / 3:.2f}")
            
            if rec.complementary_datasets:
                print(f"   Complementary: {', '.join(rec.complementary_datasets)}")
        
        # Show execution plan
        print(f"\n=== Execution Plan ===")
        for step in response.execution_plan:
            print(f"✓ {step}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await agent.cmr_api_agent.close()


async def example_temporal_analysis():
    """Example: Temporal analysis with specific constraints."""
    
    print("\n\n=== Temporal Analysis Example ===")
    
    agent = CMRAgentGraph()
    
    # Query with specific temporal and spatial constraints
    query = ("Show me MODIS vegetation index data for the Amazon rainforest "
             "from 2000 to 2020 to analyze deforestation trends.")
    
    print(f"Query: {query}")
    print("\nProcessing...")
    
    try:
        response = await agent.process_query(query)
        
        print(f"\nQuery ID: {response.query_id}")
        print(f"Success: {response.success}")
        print(f"Intent: {response.intent}")
        
        print(f"\n=== Summary ===")
        print(response.summary)
        
        # Focus on temporal coverage analysis
        for result in response.analysis_results:
            if result.analysis_type == "coverage_analysis":
                print(f"\n=== Temporal Coverage Analysis ===")
                temporal_stats = result.results.get("temporal", {})
                if temporal_stats:
                    print(f"Earliest data: {temporal_stats.get('earliest_data', 'N/A')}")
                    print(f"Collections with temporal info: {temporal_stats.get('collection_count_with_temporal', 0)}")
                    print(f"Total span (days): {temporal_stats.get('total_temporal_span_days', 0)}")
            
            elif result.analysis_type == "temporal_gap_analysis":
                print(f"\n=== Gap Analysis ===")
                gap_stats = result.results.get("statistics", {})
                if gap_stats:
                    print(f"Total gaps found: {gap_stats.get('total_gaps', 0)}")
                    print(f"Average gap duration: {gap_stats.get('average_gap_days', 0):.1f} days")
                    print(f"Longest gap: {gap_stats.get('longest_gap_days', 0)} days")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await agent.cmr_api_agent.close()


def show_system_requirements():
    """Display system requirements and setup instructions."""
    
    print("=== NASA CMR AI Agent - System Requirements ===")
    print("""
    Environment Variables Required:
    - OPENAI_API_KEY or ANTHROPIC_API_KEY (for LLM services)
    - Optional: CMR_BASE_URL (defaults to NASA CMR)
    - Optional: REDIS_URL (for caching, defaults to localhost)
    
    Installation:
    1. Install dependencies: poetry install
    2. Set up environment variables
    3. Run: python examples/basic_usage.py
    
    API Server:
    - Run: python -m nasa_cmr_agent.api.main
    - Access docs at: http://localhost:8000/docs
    - Health check: http://localhost:8000/health
    
    Key Features Demonstrated:
    - Natural language query interpretation
    - Multi-agent parallel processing  
    - NASA CMR API integration with error handling
    - Advanced dataset analysis and recommendations
    - Temporal and spatial coverage analysis
    - Gap detection and data fusion suggestions
    """)


async def main():
    """Run all examples."""
    
    # Check for required environment variables
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("WARNING: No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        print("Some features may not work properly.\n")
    
    show_system_requirements()
    
    # Run examples
    await example_precipitation_query()
    await example_urban_heat_query()  
    await example_temporal_analysis()
    
    print("\n=== Examples Complete ===")
    print("For more advanced usage, see the API documentation at /docs")


if __name__ == "__main__":
    asyncio.run(main())