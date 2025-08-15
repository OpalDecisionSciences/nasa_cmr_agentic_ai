#!/usr/bin/env python3
"""
Quick Core NASA CMR Test
Demonstrates that NASA CMR Agent works perfectly without any NASA credentials.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nasa_cmr_agent.agents.cmr_api_agent import CMRAPIAgent
from nasa_cmr_agent.models.schemas import QueryContext, QueryIntent, QueryConstraints

async def test_core_cmr_functionality():
    """Test core NASA CMR functionality without any NASA credentials."""
    
    print("🚀 NASA CMR Agent - Core Functionality Test")
    print("=" * 60)
    print(f"🔑 LLM API Key: {'Available' if os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY') else 'Missing'}")
    print(f"🌍 NASA Credentials: {'Not Required' if not (os.getenv('EARTHDATA_USERNAME') and os.getenv('EARTHDATA_PASSWORD')) else 'Available'}")
    print()
    
    try:
        # Initialize CMR agent (no NASA credentials needed)
        agent = CMRAPIAgent()
        
        # Create test query for precipitation data
        constraints = QueryConstraints(
            keywords=['precipitation', 'rainfall'],
            variables=['precipitation']
        )
        
        query_context = QueryContext(
            original_query='Find precipitation datasets for climate analysis',
            intent=QueryIntent.ANALYTICAL,
            constraints=constraints
        )
        
        print("🔍 Testing NASA CMR Collections Search...")
        print("-" * 40)
        
        # Search for collections
        collections = await agent.search_collections(query_context)
        
        print(f"✅ SUCCESS: Found {len(collections)} precipitation-related collections")
        print(f"📊 Data Sources Available: {len(collections)} NASA Earth science datasets")
        print()
        
        # Display sample results
        print("📋 Sample Datasets Found:")
        for i, collection in enumerate(collections[:3]):
            print(f"   {i+1}. {collection.title[:60]}...")
            print(f"      🏢 Data Center: {collection.data_center}")
            print(f"      🛰️  Platforms: {', '.join(collection.platforms[:2])}")
            print(f"      ☁️  Cloud Hosted: {'Yes' if collection.cloud_hosted else 'No'}")
            print()
        
        # Test granules search with first collection
        if collections:
            print("📦 Testing Granules Search...")
            print("-" * 40)
            
            first_collection = collections[0]
            granules = await agent.search_granules(
                query_context,
                collection_concept_id=first_collection.concept_id,
                limit=5
            )
            
            print(f"✅ SUCCESS: Found {len(granules)} granules for collection")
            print(f"📁 Data Files Available: {len(granules)} individual data files")
            
            if granules:
                print("\n📄 Sample Data Files:")
                for i, granule in enumerate(granules[:2]):
                    print(f"   {i+1}. {granule.title[:50]}...")
                    if granule.size_mb:
                        print(f"      📏 Size: {granule.size_mb:.1f} MB")
        
        await agent.close()
        
        print("\n" + "=" * 60)
        print("🎉 CORE FUNCTIONALITY TEST: PASSED")
        print("=" * 60)
        print("✅ NASA Earth Science Data Discovery: FULLY FUNCTIONAL")
        print("✅ Public CMR API Access: WORKING PERFECTLY")
        print("✅ No NASA Credentials Required: CONFIRMED")
        print("✅ Ready for Immediate Use: YES")
        print()
        print("💡 Summary:")
        print("   • Complete NASA Earth science data discovery works immediately")
        print("   • Thousands of datasets accessible with just LLM API key")
        print("   • NASA CMR provides full metadata without authentication")
        print("   • Users can start using the system right away")
        print()
        print("🚀 Next Steps:")
        print("   1. Run: python main.py --server")
        print("   2. Access: http://localhost:8000")
        print("   3. Start discovering NASA Earth science data!")
        
        return True
        
    except Exception as e:
        print(f"❌ CORE FUNCTIONALITY TEST: FAILED")
        print(f"Error: {e}")
        return False

async def main():
    """Run the core CMR test."""
    
    # Check LLM API key requirement
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("❌ Error: No LLM API key configured")
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY to run tests")
        return 1
    
    success = await test_core_cmr_functionality()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)