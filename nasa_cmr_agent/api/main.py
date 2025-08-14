import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
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
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
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
            # Return streaming response
            return StreamingResponse(
                _stream_query_processing(request.query),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-cache"}
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


# Streaming support
async def _stream_query_processing(query: str):
    """Stream query processing results as they become available."""
    try:
        # This would implement real streaming of agent results
        # For now, return the complete response in streaming format
        response = await agent_graph.process_query(query)
        
        # Stream agent responses as they complete
        for agent_response in response.agent_responses:
            yield f"data: {json.dumps(agent_response.model_dump())}\n\n"
        
        # Stream final response
        yield f"data: {json.dumps(response.model_dump())}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_data = {"error": str(e), "type": "processing_error"}
        yield f"data: {json.dumps(error_data)}\n\n"


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