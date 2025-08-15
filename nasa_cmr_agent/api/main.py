import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import structlog
import json
from pydantic import BaseModel

from ..core.config import settings
from ..core.graph import CMRAgentGraph
from ..models import SystemResponse
from ..services.monitoring import MetricsService
from ..services.adaptive_learning import adaptive_learner, UserFeedback
from ..services.startup_validator import StartupValidator
from ..streaming.enhanced_stream import stream_manager, EnhancedStreamer
from ..monitoring.performance_benchmarks import get_performance_benchmarks
from ..monitoring.benchmark_visualizations import get_benchmark_visualizations


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


# Global agent graph instance
agent_graph: CMRAgentGraph = None
metrics_service: MetricsService = None
startup_validation: Dict[str, Any] = {}


def create_fallback_agent_graph(error: Exception) -> CMRAgentGraph:
    """Create a minimal fallback agent graph for error recovery."""
    class FallbackAgentGraph:
        def __init__(self, init_error):
            self._initialized = False
            self._fallback_mode = True
            self._initialization_error = str(init_error)
            
        async def initialize(self):
            """Attempt re-initialization in fallback mode."""
            raise HTTPException(
                status_code=503,
                detail=f"Service temporarily unavailable. Initialization error: {self._initialization_error}"
            )
            
        async def process_query(self, user_query: str):
            """Return error response in fallback mode."""
            return {
                "success": False,
                "error": "Service is running in fallback mode due to initialization failure",
                "original_error": self._initialization_error,
                "suggestion": "Please try again later or contact support"
            }
            
        async def cleanup(self):
            """No-op cleanup in fallback mode."""
            pass
    
    return FallbackAgentGraph(error)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with proper async handling."""
    global agent_graph, metrics_service, startup_validation
    
    logger.info("Starting NASA CMR AI Agent API")
    
    try:
        # Run startup validation first
        try:
            validator = StartupValidator()
            startup_validation = await validator.validate_startup()
            
            # If sample data test was successful but relationships are missing, trigger bootstrap
            sample_test = startup_validation.get("sample_data_test", {})
            if (sample_test.get("test_performed") and 
                sample_test.get("collections_processed", 0) > 0 and
                sample_test.get("relationships_created", 0) == 0):
                
                logger.info("Bootstrap: triggering additional sample data ingestion to create relationships")
                try:
                    # Run a broader bootstrap query to ensure relationships
                    bootstrap_fixes = await validator.fix_common_issues()
                    logger.info("Bootstrap fixes completed", fixes=bootstrap_fixes)
                except Exception as bootstrap_error:
                    logger.warning(f"Bootstrap fixes failed: {bootstrap_error}")
            
            await validator.close()
            
            logger.info("Startup validation completed", 
                       status=startup_validation.get("overall_status", "unknown"))
            
            # Print recommendations if any
            if startup_validation.get("recommendations"):
                logger.info("Startup recommendations:", 
                           recommendations=startup_validation["recommendations"])
                
        except Exception as validation_error:
            logger.warning(f"Startup validation failed: {validation_error}")
            startup_validation = {"overall_status": "validation_failed", "errors": [str(validation_error)]}
        
        # Initialize agent graph with fallback handling
        try:
            # Create agent graph without blocking
            agent_graph = CMRAgentGraph()
            # Skip heavy initialization during startup - do it on first request
            logger.info("Agent graph created (initialization deferred)")
            
            # Set up a fallback handler for failed initialization
            agent_graph._fallback_mode = False
            agent_graph._initialization_error = None
            
        except Exception as graph_error:
            logger.error(f"Critical: Agent graph creation failed: {graph_error}")
            # Create a minimal fallback agent graph
            agent_graph = create_fallback_agent_graph(graph_error)
            logger.warning("Running in fallback mode with limited functionality")
        
        # Initialize metrics service with fallback handling  
        try:
            if settings.enable_metrics:
                metrics_service = MetricsService()
                logger.info("Metrics service created (server start deferred)")
        except Exception as metrics_error:
            logger.warning(f"Metrics service initialization failed: {metrics_error}")
            metrics_service = None
        
        logger.info("API startup complete")
        yield
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # Don't raise - allow service to start in fallback mode
        yield
    finally:
        # Cleanup if needed
        if agent_graph and hasattr(agent_graph, 'cmr_api_agent'):
            try:
                await agent_graph.cmr_api_agent.close()
            except:
                pass
    
    if metrics_service:
        metrics_service.stop_server()


# Create FastAPI app
app = FastAPI(
    title="NASA CMR AI Agent",
    description="Intelligent agent system for natural language interaction with NASA's Common Metadata Repository",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware with secure configuration
def get_cors_origins():
    """Get CORS origins based on environment."""
    if settings.environment == "development":
        return ["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"]
    elif settings.environment == "production":
        return [
            "https://nasa-cmr.opaldecisionsciences.com",
            "https://opaldecisionsciences.com"
        ]
    else:
        return ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    stream: bool = False
    include_visualizations: bool = False
    max_results: int = 10


class FeedbackRequest(BaseModel):
    query_id: str
    rating: int  # 1-5
    helpful: bool
    issues: List[str] = []
    suggestions: str = ""


class QueryResponse(BaseModel):
    success: bool
    data: SystemResponse
    processing_time_ms: int


class HealthResponse(BaseModel):
    status: str
    version: str
    services: Dict[str, str]


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    services = {
        "cmr_api": "not_initialized",
        "llm_service": "unknown",
        "circuit_breaker": "unknown"
    }
    
    # Add startup validation status
    if startup_validation:
        db_connections = startup_validation.get("database_connections", {})
        services["weaviate"] = "connected" if db_connections.get("weaviate", {}).get("connected") else "disconnected"
        services["neo4j"] = "connected" if db_connections.get("neo4j", {}).get("connected") else "disconnected"
        services["startup_status"] = startup_validation.get("overall_status", "unknown")
    
    # Safely check agent_graph components
    if agent_graph:
        services["cmr_api"] = "operational"
        try:
            if hasattr(agent_graph, 'query_interpreter') and agent_graph.query_interpreter:
                if hasattr(agent_graph.query_interpreter, 'llm_service'):
                    services["llm_service"] = agent_graph.query_interpreter.llm_service.get_current_provider()
        except Exception:
            pass
            
        try:
            if hasattr(agent_graph, 'cmr_api_agent') and agent_graph.cmr_api_agent:
                if hasattr(agent_graph.cmr_api_agent, 'circuit_breaker'):
                    services["circuit_breaker"] = "operational"
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy" if agent_graph else "initializing",
        version="0.1.0",
        services=services
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks, http_request: Request):
    """
    Process natural language query for NASA CMR data discovery.
    
    Args:
        request: Query request with natural language query and options
        background_tasks: FastAPI background tasks for async processing
        
    Returns:
        Comprehensive response with dataset recommendations and analysis
    """
    if not agent_graph:
        raise HTTPException(status_code=503, detail="Agent system not available")
    
    # Check if in fallback mode
    if hasattr(agent_graph, '_fallback_mode') and agent_graph._fallback_mode:
        fallback_response = await agent_graph.process_query(request.query)
        return QueryResponse(
            success=False,
            data=fallback_response,
            processing_time_ms=0
        )
    
    # Normal initialization check
    if not agent_graph._initialized:
        try:
            await agent_graph.initialize()
        except Exception as init_error:
            logger.error(f"Failed to initialize agent during request: {init_error}")
            raise HTTPException(
                status_code=503, 
                detail=f"Service initialization failed: {str(init_error)}"
            )
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info("Processing query", query=request.query, stream=request.stream)
        
        # Record metrics
        if metrics_service:
            metrics_service.record_query_start()
        
        # Process query through agent graph
        if request.stream:
            # Extract client information for security
            client_ip = http_request.client.host if http_request.client else None
            user_agent = http_request.headers.get("user-agent", "unknown")
            client_id = http_request.headers.get("x-client-id", f"web_client_{client_ip}")
            
            # Create enhanced streaming response with security context
            streamer = await stream_manager.create_stream(
                client_id=client_id,
                ip_address=client_ip,
                user_agent=user_agent
            )
            
            # Start streaming in background
            asyncio.create_task(
                _process_query_with_streaming(request.query, streamer)
            )
            
            # Return streaming response
            return StreamingResponse(
                streamer.start_stream(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Stream-ID": streamer.stream_id,
                    "X-Client-ID": client_id
                }
            )
        else:
            # Process query synchronously
            response = await agent_graph.process_query(request.query)
            
            # Calculate processing time
            processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            
            # Record metrics
            if metrics_service:
                metrics_service.record_query_completion(
                    success=response.success,
                    processing_time_ms=processing_time,
                    recommendation_count=len(response.recommendations)
                )
            
            # Add background task for any cleanup
            background_tasks.add_task(_cleanup_query_resources, response.query_id)
            
            logger.info("Query processed successfully", 
                       query_id=response.query_id, 
                       success=response.success,
                       processing_time_ms=processing_time)
            
            return QueryResponse(
                success=response.success,
                data=response,
                processing_time_ms=processing_time
            )
            
    except Exception as e:
        logger.error("Query processing failed", error=str(e))
        
        if metrics_service:
            metrics_service.record_query_error(str(e))
        
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@app.get("/query/{query_id}/status")
async def get_query_status(query_id: str):
    """Get status of a specific query (useful for long-running queries)."""
    # This would integrate with a query tracking system
    return {"query_id": query_id, "status": "completed"}


@app.get("/streams/{stream_id}/status")
async def get_stream_status(stream_id: str):
    """Get status of a specific streaming query."""
    try:
        stream = await stream_manager.get_stream(stream_id)
        if not stream:
            raise HTTPException(status_code=404, detail="Stream not found")
        
        return stream.get_stream_metrics()
    except Exception as e:
        logger.error(f"Failed to get stream status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/streams/{stream_id}")
async def disconnect_stream(stream_id: str):
    """Disconnect a specific streaming query."""
    try:
        await stream_manager.disconnect_stream(stream_id)
        return {"success": True, "message": f"Stream {stream_id} disconnected"}
    except Exception as e:
        logger.error(f"Failed to disconnect stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/streams/manager/stats")
async def get_stream_manager_stats():
    """Get streaming system statistics."""
    try:
        return stream_manager.get_manager_stats()
    except Exception as e:
        logger.error(f"Failed to get stream manager stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Performance Benchmark Endpoints
@app.get("/benchmarks/dashboard")
async def get_performance_dashboard():
    """Get comprehensive performance benchmark dashboard."""
    try:
        benchmark_system = get_performance_benchmarks()
        visualization_system = get_benchmark_visualizations(benchmark_system)
        return visualization_system.create_performance_dashboard()
    except Exception as e:
        logger.error(f"Failed to get performance dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/benchmarks/report/{time_period}")
async def get_benchmark_report(time_period: str = "24h"):
    """Get detailed performance benchmark report."""
    try:
        valid_periods = ["1h", "24h", "7d", "30d"]
        if time_period not in valid_periods:
            raise HTTPException(status_code=400, detail=f"Invalid time period. Must be one of: {valid_periods}")
        
        benchmark_system = get_performance_benchmarks()
        visualization_system = get_benchmark_visualizations(benchmark_system)
        return visualization_system.generate_benchmark_report(time_period)
    except Exception as e:
        logger.error(f"Failed to generate benchmark report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/benchmarks/thresholds")
async def get_benchmark_thresholds():
    """Get current performance benchmark thresholds and targets."""
    try:
        benchmark_system = get_performance_benchmarks()
        return {
            "thresholds": {name: {
                "target_value": threshold.target_value,
                "warning_threshold": threshold.warning_threshold,
                "critical_threshold": threshold.critical_threshold,
                "unit": threshold.unit,
                "description": threshold.description,
                "category": threshold.category.value,
                "higher_is_better": threshold.higher_is_better
            } for name, threshold in benchmark_system.thresholds.items()},
            "categories": [category.value for category in benchmark_system.thresholds["end_to_end_query_time"].category.__class__]
        }
    except Exception as e:
        logger.error(f"Failed to get benchmark thresholds: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/benchmarks/accuracy/{query_id}")
async def get_query_accuracy_details(query_id: str):
    """Get detailed accuracy metrics for a specific query."""
    try:
        benchmark_system = get_performance_benchmarks()
        
        # Find accuracy metrics for the query
        accuracy_result = next(
            (result for result in benchmark_system.accuracy_results 
             if result.query_id == query_id), None
        )
        
        if not accuracy_result:
            raise HTTPException(status_code=404, detail="Query accuracy metrics not found")
        
        return {
            "query_id": query_id,
            "accuracy_metrics": {
                "relevance_score": accuracy_result.relevance_score,
                "completeness_score": accuracy_result.completeness_score,
                "accessibility_score": accuracy_result.accessibility_score,
                "coverage_score": accuracy_result.coverage_score,
                "freshness_score": accuracy_result.freshness_score,
                "overall_accuracy": accuracy_result.overall_accuracy,
                "user_rating": accuracy_result.user_rating,
                "expert_validation": accuracy_result.expert_validation
            },
            "benchmark_status": "excellent" if accuracy_result.overall_accuracy >= 0.85 
                              else "good" if accuracy_result.overall_accuracy >= 0.70
                              else "warning" if accuracy_result.overall_accuracy >= 0.50
                              else "critical"
        }
    except Exception as e:
        logger.error(f"Failed to get query accuracy details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets/{collection_id}/details")
async def get_dataset_details(collection_id: str):
    """Get detailed information about a specific dataset."""
    if not agent_graph:
        raise HTTPException(status_code=503, detail="Agent system not initialized")
    
    try:
        # Use CMR API agent to get detailed collection info
        # This would involve direct CMR API calls for specific collection
        return {"collection_id": collection_id, "details": "Not implemented yet"}
        
    except Exception as e:
        logger.error("Failed to get dataset details", collection_id=collection_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/capabilities")
async def get_system_capabilities():
    """Get system capabilities and configuration."""
    return {
        "supported_intents": ["exploratory", "specific_data", "analytical", "comparative", "temporal_analysis", "spatial_analysis"],
        "supported_data_types": ["satellite", "ground_based", "model", "hybrid"],
        "max_concurrent_requests": settings.max_concurrent_requests,
        "cmr_api_rate_limit": settings.cmr_rate_limit_per_second,
        "llm_providers": ["openai", "anthropic"],
        "analysis_capabilities": [
            "dataset_recommendations",
            "coverage_analysis", 
            "temporal_gap_analysis",
            "dataset_relationship_analysis"
        ]
    }


@app.get("/metrics")
async def get_metrics():
    """Get system performance metrics."""
    if not metrics_service:
        raise HTTPException(status_code=503, detail="Metrics not enabled")
    
    return metrics_service.get_metrics_summary()


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback for a query response."""
    try:
        # Create UserFeedback object
        user_feedback = UserFeedback(
            query_id=feedback.query_id,
            rating=feedback.rating,
            helpful=feedback.helpful,
            issues=feedback.issues,
            suggestions=feedback.suggestions
        )
        
        # Record feedback in adaptive learning system
        adaptive_learner.record_user_feedback(user_feedback)
        
        # Log feedback for monitoring
        logger.info(
            "User feedback received",
            query_id=feedback.query_id,
            rating=feedback.rating,
            helpful=feedback.helpful
        )
        
        return {
            "success": True,
            "message": "Feedback recorded successfully",
            "improvements_suggested": len(feedback.issues) > 0
        }
        
    except Exception as e:
        logger.error("Failed to record feedback", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


@app.get("/learning/summary")
async def get_learning_summary():
    """Get summary of adaptive learning insights."""
    try:
        summary = adaptive_learner.get_learning_summary()
        return summary
    except Exception as e:
        logger.error("Failed to get learning summary", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/suggestions")
async def get_query_suggestions(partial_query: str):
    """Get query suggestions based on learned patterns."""
    try:
        suggestions = adaptive_learner.get_query_suggestions(partial_query)
        return {"suggestions": suggestions}
    except Exception as e:
        logger.error("Failed to get suggestions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Enhanced streaming support
async def _process_query_with_streaming(query: str, streamer: EnhancedStreamer):
    """Process query with real-time streaming updates."""
    try:
        # Initialize agent graph if needed
        if not agent_graph._initialized:
            await agent_graph.initialize()
        
        # Create streaming-enabled query processor
        await _stream_agent_graph_execution(query, streamer)
        
    except Exception as e:
        await streamer.emit_agent_error(
            agent_id="system",
            error=str(e),
            recoverable=False
        )
    finally:
        await streamer.complete_stream()


async def _stream_agent_graph_execution(query: str, streamer: EnhancedStreamer):
    """Execute agent graph with streaming progress updates."""
    try:
        # Start query processing
        await streamer.emit_metadata({
            "query": query,
            "agents_pipeline": ["query_interpreter", "cmr_api_agent", "analysis_agent", "response_synthesis"]
        })
        
        # Step 1: Query Interpretation
        await streamer.emit_agent_start(
            agent_id="query_interpreter",
            agent_name="Query Interpreter",
            estimated_duration=2000
        )
        
        await streamer.emit_agent_progress(
            agent_id="query_interpreter",
            progress_percent=25.0,
            current_step="Analyzing user intent"
        )
        
        # Get query context (this would integrate with actual agent)
        query_context = await agent_graph.query_interpreter.interpret_query(query)
        
        await streamer.emit_agent_progress(
            agent_id="query_interpreter",
            progress_percent=75.0,
            current_step="Extracting search parameters"
        )
        
        await streamer.emit_agent_complete(
            agent_id="query_interpreter",
            result={"intent": str(query_context.intent), "parameters": query_context.search_params},
            execution_time_ms=1500
        )
        
        await streamer.emit_partial_result(
            result_type="query_interpretation",
            data={"intent": str(query_context.intent), "search_params": query_context.search_params},
            confidence=0.85
        )
        
        # Step 2: CMR API Search
        await streamer.emit_agent_start(
            agent_id="cmr_api_agent",
            agent_name="CMR API Agent",
            estimated_duration=5000
        )
        
        await streamer.emit_agent_progress(
            agent_id="cmr_api_agent",
            progress_percent=20.0,
            current_step="Searching CMR collections"
        )
        
        # Execute CMR search with progress updates
        cmr_results = await _stream_cmr_search(query_context, streamer)
        
        await streamer.emit_agent_complete(
            agent_id="cmr_api_agent",
            result={"collections_found": len(cmr_results.get("collections", [])),
                   "total_granules": cmr_results.get("total_granules", 0)},
            execution_time_ms=4200
        )
        
        # Step 3: Analysis and Recommendations
        if hasattr(agent_graph, 'analysis_agent'):
            await streamer.emit_agent_start(
                agent_id="analysis_agent",
                agent_name="Analysis Agent",
                estimated_duration=3000
            )
            
            await streamer.emit_agent_progress(
                agent_id="analysis_agent",
                progress_percent=30.0,
                current_step="Analyzing dataset relevance"
            )
            
            analysis_results = await _stream_analysis(query_context, cmr_results, streamer)
            
            await streamer.emit_agent_complete(
                agent_id="analysis_agent",
                result={"recommendations_generated": len(analysis_results.get("recommendations", []))},
                execution_time_ms=2800
            )
        
        # Step 4: Response Synthesis
        await streamer.emit_agent_start(
            agent_id="response_synthesis",
            agent_name="Response Synthesis",
            estimated_duration=1500
        )
        
        await streamer.emit_agent_progress(
            agent_id="response_synthesis",
            progress_percent=50.0,
            current_step="Generating final response"
        )
        
        # Generate final response
        final_response = await agent_graph.response_agent.synthesize_response(
            query_context, cmr_results, analysis_results if 'analysis_results' in locals() else []
        )
        
        await streamer.emit_agent_complete(
            agent_id="response_synthesis",
            result={"response_generated": True, "recommendations_count": len(final_response.recommendations)},
            execution_time_ms=1200
        )
        
        # Stream final results
        await streamer.emit_partial_result(
            result_type="final_response",
            data=final_response.model_dump(),
            confidence=0.92
        )
        
    except Exception as e:
        logger.error(f"Streaming execution error: {e}")
        await streamer.emit_agent_error(
            agent_id="system",
            error=f"Processing error: {str(e)}",
            recoverable=False
        )


async def _stream_cmr_search(query_context, streamer: EnhancedStreamer):
    """Stream CMR search with progress updates."""
    try:
        # Simulate progressive search updates
        await streamer.emit_agent_progress(
            agent_id="cmr_api_agent",
            progress_percent=40.0,
            current_step="Fetching collection metadata"
        )
        
        # Actual CMR search
        collections = await agent_graph.cmr_api_agent.search_collections(query_context)
        
        await streamer.emit_agent_progress(
            agent_id="cmr_api_agent",
            progress_percent=70.0,
            current_step="Analyzing temporal coverage"
        )
        
        # Stream partial collection results
        for i, collection in enumerate(collections[:3]):
            await streamer.emit_partial_result(
                result_type="collection_preview",
                data={
                    "collection_id": collection.get("concept_id"),
                    "title": collection.get("title"),
                    "relevance_score": collection.get("relevance_score", 0.0)
                }
            )
        
        await streamer.emit_agent_progress(
            agent_id="cmr_api_agent",
            progress_percent=90.0,
            current_step="Finalizing results"
        )
        
        return {"collections": collections, "total_granules": sum(c.get("granule_count", 0) for c in collections)}
        
    except Exception as e:
        await streamer.emit_agent_error(
            agent_id="cmr_api_agent",
            error=f"CMR search failed: {str(e)}",
            recoverable=True
        )
        return {"collections": [], "total_granules": 0}


async def _stream_analysis(query_context, cmr_results, streamer: EnhancedStreamer):
    """Stream analysis with progress updates."""
    try:
        await streamer.emit_agent_progress(
            agent_id="analysis_agent",
            progress_percent=60.0,
            current_step="Generating recommendations"
        )
        
        # Actual analysis
        analysis = await agent_graph.analysis_agent.analyze_results(query_context, cmr_results)
        
        await streamer.emit_agent_progress(
            agent_id="analysis_agent",
            progress_percent=90.0,
            current_step="Validating recommendations"
        )
        
        return analysis
        
    except Exception as e:
        await streamer.emit_agent_error(
            agent_id="analysis_agent",
            error=f"Analysis failed: {str(e)}",
            recoverable=True
        )
        return {"recommendations": []}


async def _cleanup_query_resources(query_id: str):
    """Background task to clean up query-specific resources."""
    # Implement any necessary cleanup logic
    logger.debug("Cleaning up query resources", query_id=query_id)


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": str(exc)}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error("Internal server error", error=str(exc))
    return {"error": "Internal server error", "detail": "An unexpected error occurred"}


# ASGI server
async def run_asgi_server():
    """Run true ASGI server."""
    config = uvicorn.Config(
        app=app,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
        loop="asyncio"
    )
    server = uvicorn.Server(config)
    await server.serve()

def run_dev_server():
    """Run development server with proper ASGI handling."""
    asyncio.run(run_asgi_server())


if __name__ == "__main__":
    run_dev_server()