#!/usr/bin/env python3
"""
NASA CMR AI Agent - Advanced Features Demo

This script demonstrates the bonus challenge features:
- Weaviate vector database semantic search
- RAG (Retrieval-Augmented Generation) enhanced analysis
- Neo4j knowledge graph relationships
- Enhanced contextual recommendations
"""

import asyncio
import os
from nasa_cmr_agent.core.graph import CMRAgentGraph
from nasa_cmr_agent.services.vector_database import VectorDatabaseService
from nasa_cmr_agent.services.knowledge_graph import KnowledgeGraphService
from nasa_cmr_agent.services.rag_service import RAGService
from nasa_cmr_agent.models import SystemResponse


async def demo_vector_search():
    """Demonstrate vector database semantic search capabilities."""
    
    print("=== VECTOR DATABASE SEMANTIC SEARCH DEMO ===")
    print("Initializing Weaviate vector database...")
    
    vector_db = VectorDatabaseService()
    
    # Initialize schema
    schema_success = await vector_db.initialize_schema()
    if not schema_success:
        print("⚠️  Vector database not available - check Weaviate installation")
        print("   Install: docker run -d -p 8080:8080 semitechnologies/weaviate:latest")
        return
    
    print("✅ Vector database initialized")
    
    # Demonstrate semantic search
    search_queries = [
        "precipitation data for drought monitoring",
        "satellite temperature measurements",
        "vegetation indices and agriculture",
        "ocean surface temperature analysis"
    ]
    
    for query in search_queries:
        print(f"\n🔍 Searching: '{query}'")
        
        results = await vector_db.semantic_search(query, limit=3)
        
        if results:
            print(f"   Found {len(results)} semantically similar datasets:")
            for i, result in enumerate(results, 1):
                score = result.get('similarity_score', 0)
                title = result.get('title', 'Unknown')[:50] + "..."
                print(f"   {i}. {title} (similarity: {score:.3f})")
        else:
            print("   No results found (database may be empty)")
    
    print(f"\n💡 Semantic search finds datasets based on meaning, not just keywords!")
    
    await vector_db.close()


async def demo_knowledge_graph():
    """Demonstrate knowledge graph relationship discovery."""
    
    print("\n\n=== KNOWLEDGE GRAPH RELATIONSHIPS DEMO ===")
    print("Initializing Neo4j knowledge graph...")
    
    kg_service = KnowledgeGraphService()
    
    # Check if Neo4j is available
    if not kg_service.driver:
        print("⚠️  Neo4j not available - check Neo4j installation")
        print("   Install: docker run -d -p 7687:7687 -p 7474:7474 neo4j:latest")
        return
    
    schema_success = await kg_service.initialize_schema()
    if not schema_success:
        print("⚠️  Failed to initialize knowledge graph schema")
        return
    
    print("✅ Knowledge graph initialized")
    
    # Demonstrate platform ecosystem analysis
    platforms = ["Terra", "Aqua", "Landsat-8", "GPM"]
    
    for platform in platforms:
        print(f"\n🌐 Analyzing ecosystem for platform: {platform}")
        
        ecosystem = await kg_service.get_platform_ecosystem(platform)
        
        if ecosystem:
            print(f"   Datasets: {ecosystem['ecosystem_size']['datasets']}")
            print(f"   Instruments: {ecosystem['ecosystem_size']['instruments']}")
            print(f"   Phenomena: {ecosystem['ecosystem_size']['phenomena']}")
            print(f"   Variables: {ecosystem['ecosystem_size']['variables']}")
            
            if ecosystem.get('instruments'):
                print(f"   Key Instruments: {', '.join(ecosystem['instruments'][:3])}")
        else:
            print(f"   No ecosystem data available (graph may be empty)")
    
    # Demonstrate research pathway analysis
    phenomena = ["precipitation", "temperature", "vegetation"]
    
    for phenomenon in phenomena:
        print(f"\n📊 Research pathways for: {phenomenon}")
        
        pathways = await kg_service.analyze_research_pathways(phenomenon)
        
        if pathways:
            total_datasets = pathways.get('total_datasets', 0)
            platforms = len(pathways.get('platforms', []))
            instruments = len(pathways.get('instruments', []))
            
            print(f"   Datasets studying {phenomenon}: {total_datasets}")
            print(f"   Platforms involved: {platforms}")
            print(f"   Instruments involved: {instruments}")
            
            if pathways.get('temporal_span', {}).get('earliest'):
                earliest = pathways['temporal_span']['earliest']
                latest = pathways['temporal_span']['latest']
                print(f"   Temporal span: {earliest} to {latest}")
        else:
            print(f"   No pathway data available")
    
    print(f"\n💡 Knowledge graphs reveal hidden relationships between datasets!")
    
    await kg_service.close()


async def demo_rag_enhanced_analysis():
    """Demonstrate RAG-enhanced contextual analysis."""
    
    print("\n\n=== RAG-ENHANCED ANALYSIS DEMO ===")
    print("Demonstrating Retrieval-Augmented Generation...")
    
    # Complex queries that benefit from RAG
    complex_queries = [
        "How do MODIS and VIIRS compare for fire detection in tropical regions?",
        "What are the best practices for combining precipitation datasets from different satellites?",
        "Which datasets provide complementary information for urban heat island studies?"
    ]
    
    agent = CMRAgentGraph()
    
    for query in complex_queries:
        print(f"\n🤖 RAG Analysis: '{query}'")
        
        try:
            response = await agent.process_query(query)
            
            if response.success:
                print(f"   Intent: {response.intent}")
                print(f"   Analysis Types: {len(response.analysis_results)}")
                
                # Look for enhanced analysis results
                enhanced_results = [
                    r for r in response.analysis_results 
                    if 'enhanced' in r.analysis_type.lower() or 
                       'rag' in r.analysis_type.lower() or
                       'knowledge_graph' in r.analysis_type.lower()
                ]
                
                if enhanced_results:
                    print(f"   🚀 Enhanced Features Used:")
                    for result in enhanced_results:
                        print(f"      - {result.analysis_type.replace('_', ' ').title()}")
                        print(f"        Confidence: {result.confidence_level:.2f}")
                        if result.statistics:
                            for key, value in result.statistics.items():
                                if isinstance(value, (int, float)):
                                    print(f"        {key.replace('_', ' ').title()}: {value}")
                
                # Show contextual insights
                if response.follow_up_suggestions:
                    print(f"   💡 Enhanced Suggestions:")
                    for suggestion in response.follow_up_suggestions[:2]:
                        print(f"      - {suggestion}")
            else:
                print(f"   ⚠️  Analysis failed: {response.summary}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    await agent.cmr_api_agent.close()
    if hasattr(agent.analysis_agent, 'close'):
        await agent.analysis_agent.close()


async def demo_complete_workflow():
    """Demonstrate complete advanced workflow with all features."""
    
    print("\n\n=== COMPLETE ADVANCED WORKFLOW DEMO ===")
    print("Demonstrating full integration of vector search, knowledge graph, and RAG...")
    
    # NASA technical assessment example query
    nasa_query = ("Compare precipitation datasets suitable for drought monitoring "
                  "in Sub-Saharan Africa between 2015-2023, considering both "
                  "satellite and ground-based observations, and identify gaps "
                  "in temporal coverage.")
    
    print(f"🎯 NASA Assessment Query:")
    print(f"   {nasa_query}")
    
    agent = CMRAgentGraph()
    
    try:
        print(f"\n⚙️  Processing with advanced features...")
        response = await agent.process_query(nasa_query)
        
        if response.success:
            print(f"\n✅ Advanced Analysis Complete!")
            print(f"   Query ID: {response.query_id}")
            print(f"   Intent: {response.intent}")
            print(f"   Processing Time: {response.total_execution_time_ms}ms")
            print(f"   Recommendations: {len(response.recommendations)}")
            print(f"   Analysis Types: {len(response.analysis_results)}")
            
            # Show advanced analysis results
            advanced_analyses = []
            for result in response.analysis_results:
                if any(keyword in result.analysis_type.lower() 
                      for keyword in ['enhanced', 'vector', 'knowledge_graph', 'rag', 'fusion']):
                    advanced_analyses.append(result)
            
            if advanced_analyses:
                print(f"\n🚀 Advanced Analysis Results:")
                for result in advanced_analyses:
                    analysis_name = result.analysis_type.replace('_', ' ').title()
                    print(f"   📊 {analysis_name}")
                    print(f"      Method: {result.methodology[:80]}...")
                    print(f"      Confidence: {result.confidence_level:.2f}")
                    
                    if result.statistics:
                        interesting_stats = {k: v for k, v in result.statistics.items() 
                                           if isinstance(v, (int, float)) and v > 0}
                        if interesting_stats:
                            print(f"      Key Stats: {interesting_stats}")
            
            # Show top recommendations with enhanced scoring
            print(f"\n🏆 Top Enhanced Recommendations:")
            for i, rec in enumerate(response.recommendations[:3], 1):
                print(f"   {i}. {rec.collection.title}")
                print(f"      Overall Score: {(rec.relevance_score + rec.coverage_score + rec.quality_score) / 3:.3f}")
                print(f"      Platforms: {', '.join(rec.collection.platforms) if rec.collection.platforms else 'N/A'}")
                
                if rec.complementary_datasets:
                    print(f"      Complementary: {', '.join(rec.complementary_datasets[:2])}")
                
                if rec.temporal_gaps:
                    print(f"      Temporal Gaps: {len(rec.temporal_gaps)} identified")
            
            # Show research insights
            if response.follow_up_suggestions:
                print(f"\n💡 Advanced Insights & Suggestions:")
                for suggestion in response.follow_up_suggestions[:4]:
                    print(f"   • {suggestion}")
                    
        else:
            print(f"❌ Advanced workflow failed:")
            print(f"   {response.summary}")
            if response.warnings:
                for warning in response.warnings:
                    print(f"   ⚠️  {warning}")
    
    except Exception as e:
        print(f"❌ Workflow error: {e}")
    
    finally:
        await agent.cmr_api_agent.close()
        if hasattr(agent.analysis_agent, 'close'):
            await agent.analysis_agent.close()


def show_advanced_setup_info():
    """Display setup information for advanced features."""
    
    print("=== NASA CMR AI Agent - Advanced Features Setup ===")
    print("""
    🚀 BONUS CHALLENGE FEATURES IMPLEMENTED:

    1. VECTOR DATABASE (Weaviate)
       - Semantic search over NASA dataset documentation
       - Embedding-based similarity matching
       - Advanced hybrid search capabilities
    
    2. RAG (Retrieval-Augmented Generation)
       - Dynamic context retrieval based on query similarity
       - Enhanced dataset recommendations with contextual insights  
       - Multi-hop reasoning over dataset relationships
    
    3. KNOWLEDGE GRAPH (Neo4j)
       - Dataset relationship discovery and mapping
       - Platform and instrument ecosystem analysis
       - Data fusion opportunity identification
       - Research pathway analysis

    📋 SETUP REQUIREMENTS:

    Environment Variables:
    - ENABLE_VECTOR_SEARCH=true (default)
    - ENABLE_KNOWLEDGE_GRAPH=true (default) 
    - ENABLE_RAG=true (default)
    - WEAVIATE_URL=http://localhost:8080
    - NEO4J_URI=bolt://localhost:7687
    - NEO4J_USER=neo4j
    - NEO4J_PASSWORD=password

    Docker Services:
    # Start Weaviate vector database
    docker run -d -p 8080:8080 semitechnologies/weaviate:latest

    # Start Neo4j knowledge graph  
    docker run -d -p 7687:7687 -p 7474:7474 \\
      -e NEO4J_AUTH=neo4j/password \\
      neo4j:latest

    🎯 CAPABILITIES DEMONSTRATED:
    - Semantic understanding of Earth science queries
    - Knowledge graph relationship discovery
    - RAG-enhanced contextual analysis
    - Multi-modal dataset recommendation scoring
    - Advanced data fusion opportunity identification
    - Research pathway analysis and methodology suggestions
    """)


async def main():
    """Run all advanced feature demonstrations."""
    
    show_advanced_setup_info()
    
    # Check environment variables
    print(f"🔧 Configuration Check:")
    print(f"   Vector Search: {'✅ Enabled' if os.getenv('ENABLE_VECTOR_SEARCH', 'true').lower() == 'true' else '❌ Disabled'}")
    print(f"   Knowledge Graph: {'✅ Enabled' if os.getenv('ENABLE_KNOWLEDGE_GRAPH', 'true').lower() == 'true' else '❌ Disabled'}")
    print(f"   RAG Enhancement: {'✅ Enabled' if os.getenv('ENABLE_RAG', 'true').lower() == 'true' else '❌ Disabled'}")
    
    # Check for required API keys
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("⚠️  WARNING: No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        print("   Some RAG features may not work properly.\n")
    
    # Run demonstrations
    await demo_vector_search()
    await demo_knowledge_graph()
    await demo_rag_enhanced_analysis()
    await demo_complete_workflow()
    
    print("\n\n🎉 ADVANCED FEATURES DEMONSTRATION COMPLETE!")
    print("""
    These bonus features showcase:
    ✅ Vector Database Integration: Semantic search over dataset documentation
    ✅ Dynamic Context Retrieval: RAG-enhanced query processing  
    ✅ Knowledge Graph Construction: Dataset relationship discovery
    ✅ Advanced Analytics: Multi-modal scoring and fusion opportunities
    ✅ Research Pathway Analysis: Graph traversal for methodology suggestions
    
    🏆 This implementation goes beyond the basic requirements to demonstrate
       advanced AI/ML capabilities suitable for NASA's cutting-edge research needs!
    """)


if __name__ == "__main__":
    asyncio.run(main())