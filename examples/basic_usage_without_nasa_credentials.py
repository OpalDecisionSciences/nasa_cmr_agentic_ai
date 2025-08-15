#!/usr/bin/env python3
"""
Basic usage example for NASA CMR Agent without NASA API credentials.

This example demonstrates that the system works perfectly with just an LLM API key,
providing full CMR data discovery capabilities without requiring any NASA authentication.

Required:
- OPENAI_API_KEY or ANTHROPIC_API_KEY

Optional NASA credentials (will enhance features but not required):
- EARTHDATA_USERNAME/EARTHDATA_PASSWORD
- LAADS_API_KEY
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nasa_cmr_agent.core.graph import CMRAgentGraph


async def demonstrate_basic_usage():
    """Demonstrate basic NASA CMR Agent usage without NASA credentials."""
    
    print("🚀 NASA CMR Agent - Basic Usage Demo")
    print("=" * 60)
    print("This demo shows that the system works perfectly without NASA API credentials.")
    print("You only need an OpenAI or Anthropic API key for LLM functionality.\n")
    
    # Check for required LLM API key
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    
    if not (has_openai or has_anthropic):
        print("❌ Error: Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        print("Example: export OPENAI_API_KEY=your_openai_api_key")
        return
    
    print(f"✅ LLM API Key: {'OpenAI' if has_openai else 'Anthropic'} configured")
    
    # Check NASA credentials status (optional)
    has_earthdata = bool(os.getenv("EARTHDATA_USERNAME") and os.getenv("EARTHDATA_PASSWORD"))
    has_laads = bool(os.getenv("LAADS_API_KEY"))
    
    print(f"📡 NASA Earthdata Login: {'✅ Available' if has_earthdata else '⚪ Not configured (optional)'}")
    print(f"🛰️  LAADS DAAC API Key: {'✅ Available' if has_laads else '⚪ Not configured (optional)'}")
    print(f"🔧 API Key Manager: {'⚪ Disabled (using environment variables)' if not os.getenv('ENABLE_API_KEY_MANAGER') else '✅ Enabled'}")
    
    print("\n" + "=" * 60)
    print("🔍 Running Sample Queries")
    print("=" * 60)
    
    # Initialize the agent
    agent = CMRAgentGraph()
    
    sample_queries = [
        "Find precipitation datasets for drought monitoring in Africa",
        "What MODIS datasets are available for land surface temperature analysis?",
        "Show me satellite data for studying urban heat islands"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 40)
        
        try:
            response = await agent.process_query(query)
            
            print(f"✅ Success: {response.success}")
            print(f"🎯 Intent: {response.intent}")
            print(f"⏱️  Processing Time: {response.total_execution_time_ms}ms")
            print(f"📊 Recommendations: {len(response.recommendations)}")
            
            if response.recommendations:
                print("📋 Top Recommendation:")
                top_rec = response.recommendations[0]
                print(f"   • {top_rec.collection.title}")
                print(f"   • Dataset ID: {top_rec.collection.concept_id}")
                print(f"   • Data Center: {top_rec.collection.data_center}")
                print(f"   • Relevance Score: {top_rec.relevance_score:.2f}")
            
            if response.warnings:
                print("⚠️  Warnings:")
                for warning in response.warnings:
                    print(f"   • {warning}")
            
            print(f"💡 Summary: {response.summary[:100]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("🎉 Demo Complete!")
    print("=" * 60)
    print("Key Points:")
    print("• ✅ System works perfectly with just LLM API keys")
    print("• 🔍 Full CMR data discovery capabilities available")
    print("• 📡 NASA credentials are optional and enhance features")
    print("• 🔧 API Key Manager is optional for production use")
    print("• 🚀 Ready for deployment without any NASA setup required")
    
    # Clean up
    await agent.cmr_api_agent.close()


async def demonstrate_enhanced_features():
    """Show what additional features are available with NASA credentials."""
    
    print("\n" + "=" * 60)
    print("🌟 Enhanced Features (with NASA credentials)")
    print("=" * 60)
    
    features = [
        {
            "feature": "GIOVANNI Data Analysis",
            "requires": "Earthdata Login",
            "benefit": "Interactive data analysis and visualization"
        },
        {
            "feature": "MODAPS/LAADS DAAC Access",
            "requires": "LAADS API Key",
            "benefit": "Enhanced MODIS data access and metadata"
        },
        {
            "feature": "Atmospheric Data APIs",
            "requires": "Earthdata Authentication",
            "benefit": "Specialized atmospheric instrument data (AIRS, OMI, MOPITT)"
        },
        {
            "feature": "Protected Dataset Access",
            "requires": "Earthdata Login",
            "benefit": "Access to restricted/early-release datasets"
        },
        {
            "feature": "Higher Rate Limits",
            "requires": "NASA Authentication",
            "benefit": "Faster data discovery for large queries"
        }
    ]
    
    print("Without NASA credentials, you get:")
    print("• ✅ Full CMR public dataset search")
    print("• ✅ Intelligent query processing")
    print("• ✅ Multi-criteria dataset recommendations")
    print("• ✅ Temporal and spatial analysis")
    print("• ✅ Production-ready API server")
    
    print("\nWith NASA credentials, you additionally get:")
    for feature in features:
        print(f"• 🌟 {feature['feature']}")
        print(f"     Requires: {feature['requires']}")
        print(f"     Benefit: {feature['benefit']}")
    
    print("\n📚 How to get NASA credentials:")
    print("1. Earthdata Login:")
    print("   • Visit: https://urs.earthdata.nasa.gov/")
    print("   • Create free account")
    print("   • Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD")
    
    print("\n2. LAADS DAAC API Key:")
    print("   • Visit: https://ladsweb.modaps.eosdis.nasa.gov/")
    print("   • Log in with Earthdata credentials")
    print("   • Generate API key in Account Settings")
    print("   • Set LAADS_API_KEY")


def main():
    """Main demo function."""
    print("Starting NASA CMR Agent demo...")
    asyncio.run(demonstrate_basic_usage())
    asyncio.run(demonstrate_enhanced_features())


if __name__ == "__main__":
    main()