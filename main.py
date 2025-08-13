#!/usr/bin/env python3
"""
NASA CMR AI Agent - Main Entry Point

This is the primary entry point for the NASA CMR AI Agent system.
Provides both CLI and web server interfaces for interacting with the system.
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from nasa_cmr_agent.api.main import run_dev_server
from nasa_cmr_agent.core.graph import CMRAgentGraph
from nasa_cmr_agent.core.config import settings


async def run_cli_query(query: str):
    """Run a single query via CLI interface."""
    
    print(f"NASA CMR AI Agent - Processing Query")
    print(f"Query: {query}")
    print("-" * 60)
    
    # Initialize agent
    agent = CMRAgentGraph()
    
    try:
        # Process query
        response = await agent.process_query(query)
        
        # Display results
        print(f"\nQuery ID: {response.query_id}")
        print(f"Success: {response.success}")
        print(f"Intent: {response.intent}")
        print(f"Processing Time: {response.total_execution_time_ms}ms")
        
        print(f"\n{'='*20} SUMMARY {'='*20}")
        print(response.summary)
        
        if response.recommendations:
            print(f"\n{'='*15} RECOMMENDATIONS ({len(response.recommendations)}) {'='*15}")
            for i, rec in enumerate(response.recommendations[:5], 1):
                print(f"\n{i}. {rec.collection.title}")
                print(f"   Dataset ID: {rec.collection.concept_id}")
                print(f"   Data Center: {rec.collection.data_center}")
                print(f"   Platforms: {', '.join(rec.collection.platforms) if rec.collection.platforms else 'N/A'}")
                print(f"   Relevance Score: {rec.relevance_score:.2f}")
                print(f"   Coverage Score: {rec.coverage_score:.2f}")
                print(f"   Quality Score: {rec.quality_score:.2f}")
                print(f"   Accessibility Score: {rec.accessibility_score:.2f}")
                print(f"   Reasoning: {rec.reasoning}")
                
                if rec.granule_count:
                    print(f"   Available Granules: {rec.granule_count}")
                
                if rec.temporal_gaps:
                    print(f"   Temporal Gaps: {len(rec.temporal_gaps)} identified")
                
                if rec.complementary_datasets:
                    print(f"   Complementary Datasets: {', '.join(rec.complementary_datasets)}")
        
        if response.analysis_results:
            print(f"\n{'='*15} ANALYSIS RESULTS {'='*15}")
            for result in response.analysis_results:
                print(f"\n• {result.analysis_type.replace('_', ' ').title()}")
                print(f"  Methodology: {result.methodology}")
                print(f"  Confidence: {result.confidence_level:.2f}")
                
                if result.statistics:
                    print("  Key Statistics:")
                    for key, value in result.statistics.items():
                        print(f"    - {key.replace('_', ' ').title()}: {value}")
        
        if response.warnings:
            print(f"\n{'='*20} WARNINGS {'='*20}")
            for warning in response.warnings:
                print(f"⚠️  {warning}")
        
        if response.follow_up_suggestions:
            print(f"\n{'='*15} FOLLOW-UP SUGGESTIONS {'='*15}")
            for suggestion in response.follow_up_suggestions:
                print(f"💡 {suggestion}")
        
        print(f"\n{'='*15} EXECUTION PLAN {'='*15}")
        for step in response.execution_plan:
            print(f"✓ {step}")
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        return 1
    
    finally:
        await agent.cmr_api_agent.close()
    
    return 0


def check_environment():
    """Check that required environment variables are set."""
    
    missing_vars = []
    
    # Check LLM API keys
    if not settings.openai_api_key and not settings.anthropic_api_key:
        missing_vars.append("OPENAI_API_KEY or ANTHROPIC_API_KEY")
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        print("See .env.example for template.")
        return False
    
    return True


def show_system_info():
    """Display system configuration and status."""
    
    print("NASA CMR AI Agent - System Information")
    print("=" * 50)
    print(f"Environment: {settings.environment}")
    print(f"CMR Base URL: {settings.cmr_base_url}")
    print(f"CMR Rate Limit: {settings.cmr_rate_limit_per_second} req/sec")
    print(f"Request Timeout: {settings.request_timeout}s")
    print(f"Max Concurrent: {settings.max_concurrent_requests}")
    print(f"Log Level: {settings.log_level}")
    
    print(f"\nLLM Configuration:")
    print(f"OpenAI API Key: {'✓ Set' if settings.openai_api_key else '✗ Not set'}")
    print(f"Anthropic API Key: {'✓ Set' if settings.anthropic_api_key else '✗ Not set'}")
    print(f"OpenAI Model: {settings.openai_model}")
    print(f"Anthropic Model: {settings.anthropic_model}")
    
    print(f"\nAPI Server Configuration:")
    print(f"Host: {settings.api_host}")
    print(f"Port: {settings.api_port}")
    print(f"Debug: {settings.api_debug}")
    
    print(f"\nMonitoring:")
    print(f"Metrics Enabled: {settings.enable_metrics}")
    print(f"Prometheus Port: {settings.prometheus_port}")


def main():
    """Main entry point with argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="NASA CMR AI Agent - Intelligent Earth Science Data Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run interactive query
  python main.py --query "Find precipitation data for drought monitoring in Africa 2020-2023"
  
  # Start web server
  python main.py --server
  
  # Show system information
  python main.py --info
  
  # Run example queries
  python examples/basic_usage.py
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        help="Process a natural language query via CLI",
        type=str
    )
    
    parser.add_argument(
        "--server", "-s",
        help="Start web server (default: http://localhost:8000)",
        action="store_true"
    )
    
    parser.add_argument(
        "--info", "-i",
        help="Show system configuration and status",
        action="store_true"
    )
    
    parser.add_argument(
        "--examples", "-e",
        help="Run example queries",
        action="store_true"
    )
    
    parser.add_argument(
        "--host",
        help="Server host (default: 0.0.0.0)",
        default=settings.api_host
    )
    
    parser.add_argument(
        "--port",
        help="Server port (default: 8000)",
        type=int,
        default=settings.api_port
    )
    
    args = parser.parse_args()
    
    # Show info if requested
    if args.info:
        show_system_info()
        return 0
    
    # Run examples if requested
    if args.examples:
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, 
                str(project_root / "examples" / "basic_usage.py")
            ])
            return result.returncode
        except Exception as e:
            print(f"❌ Error running examples: {e}")
            return 1
    
    # Check environment setup
    if not check_environment():
        return 1
    
    # Process CLI query
    if args.query:
        return asyncio.run(run_cli_query(args.query))
    
    # Start server
    if args.server:
        print(f"🚀 Starting NASA CMR AI Agent server...")
        print(f"   Server: http://{args.host}:{args.port}")
        print(f"   Documentation: http://{args.host}:{args.port}/docs")
        print(f"   Health Check: http://{args.host}:{args.port}/health")
        
        # Update settings with CLI args
        settings.api_host = args.host
        settings.api_port = args.port
        
        try:
            run_dev_server()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")
        except Exception as e:
            print(f"❌ Server error: {e}")
            return 1
        
        return 0
    
    # If no specific action, show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())