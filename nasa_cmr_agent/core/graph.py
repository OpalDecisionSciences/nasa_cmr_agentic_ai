from typing import Dict, Any, List
from typing_extensions import TypedDict
import asyncio
import logging
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from ..models.schemas import QueryContext, SystemResponse, AgentResponse, QueryIntent
from ..agents.query_interpreter import QueryInterpreterAgent
from ..agents.cmr_api_agent import CMRAPIAgent
from ..agents.enhanced_analysis_agent import EnhancedAnalysisAgent
from ..agents.response_agent import ResponseSynthesisAgent
from ..agents.supervisor_agent import SupervisorAgent
from ..tools.scratchpad import scratchpad_manager
from ..services.supervisor_learning_integration import supervisor_learning
from ..services.adaptive_learning import adaptive_learner
from ..core.config import settings

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State shared between agents in the LangGraph workflow."""
    messages: List[Any]
    query_context: Any
    cmr_results: Dict[str, Any]
    analysis_results: List[Any]
    agent_responses: List[Any]
    errors: List[Any]
    next_agent: Any
    original_query: str
    retry_count: int


class CMRAgentGraph:
    """
    Multi-agent workflow orchestrator using LangGraph for NASA CMR data discovery.
    
    Implements a sophisticated pipeline:
    1. Query interpretation and validation
    2. Parallel CMR API interactions
    3. Data analysis and recommendation generation
    4. Response synthesis and formatting
    """
    
    def __init__(self):
        # Only set basic attributes - defer initialization
        self._initialized = False
        self.max_retries = 2
        
    async def initialize(self):
        """Async initialization of all components."""
        if self._initialized:
            return
            
        self.query_interpreter = QueryInterpreterAgent()
        self.cmr_api_agent = CMRAPIAgent()
        
        # Use enhanced analysis agent if advanced features are enabled
        if (settings.enable_vector_search or 
            settings.enable_knowledge_graph or 
            settings.enable_rag):
            self.analysis_agent = EnhancedAnalysisAgent()
        else:
            # Fallback to basic analysis agent
            from ..agents.analysis_agent import DataAnalysisAgent
            self.analysis_agent = DataAnalysisAgent()
        
        self.response_agent = ResponseSynthesisAgent()
        self.supervisor = SupervisorAgent()
        self.scratchpad_manager = scratchpad_manager
        
        self.graph = self._build_graph()
        self._initialized = True
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("interpret_query", self._interpret_query_node)
        workflow.add_node("validate_query", self._validate_query_node)
        workflow.add_node("search_collections", self._search_collections_node)
        workflow.add_node("search_granules", self._search_granules_node)
        workflow.add_node("analyze_data", self._analyze_data_node)
        workflow.add_node("synthesize_response", self._synthesize_response_node)
        
        # Define the workflow edges
        workflow.set_entry_point("interpret_query")
        
        # Sequential flow with conditional branching
        workflow.add_edge("interpret_query", "validate_query")
        workflow.add_conditional_edges(
            "validate_query",
            self._should_continue_after_validation,
            {
                "continue": "search_collections",
                "error": "synthesize_response"
            }
        )
        
        # Parallel execution for collections and granules
        workflow.add_edge("search_collections", "search_granules")
        workflow.add_edge("search_granules", "analyze_data")
        workflow.add_edge("analyze_data", "synthesize_response")
        workflow.add_edge("synthesize_response", END)
        
        return workflow.compile()
    
    async def _interpret_query_node(self, state: AgentState) -> AgentState:
        """Node for query interpretation and decomposition."""
        try:
            messages = state["messages"]
            if not messages:
                raise ValueError("No messages provided")
            
            user_query = messages[-1].content
            query_context = await self.query_interpreter.interpret_query(user_query)
            
            # Get decomposed queries for complex comparative queries
            sub_queries = None
            if query_context.intent in [QueryIntent.COMPARATIVE, QueryIntent.ANALYTICAL]:
                try:
                    strategy, sub_queries = await self.query_interpreter.get_decomposed_queries(user_query, query_context)
                    state["decomposition_strategy"] = strategy
                    state["sub_queries"] = sub_queries
                except Exception as e:
                    logger.warning(f"Query decomposition failed: {e}")
            
            state["query_context"] = query_context
            state["agent_responses"].append(AgentResponse(
                agent_name="query_interpreter",
                status="success",
                data={
                    "query_context": query_context.model_dump(),
                    "sub_queries_count": len(sub_queries) if sub_queries else 0
                },
                execution_time_ms=0  # TODO: Add timing
            ))
            
        except Exception as e:
            state["errors"].append(f"Query interpretation failed: {str(e)}")
            state["agent_responses"].append(AgentResponse(
                agent_name="query_interpreter",
                status="error",
                error=str(e)
            ))
        
        return state
    
    async def _validate_query_node(self, state: AgentState) -> AgentState:
        """Node for query validation."""
        try:
            query_context = state.get("query_context")
            if not query_context:
                raise ValueError("No query context available")
            
            is_valid = await self.query_interpreter.validate_query(query_context)
            
            if not is_valid:
                state["errors"].append("Query validation failed")
                state["next_agent"] = "error"
            else:
                state["next_agent"] = "continue"
            
            state["agent_responses"].append(AgentResponse(
                agent_name="query_validator",
                status="success" if is_valid else "validation_failed",
                data={"is_valid": is_valid}
            ))
            
        except Exception as e:
            state["errors"].append(f"Query validation error: {str(e)}")
            state["next_agent"] = "error"
            
        return state
    
    async def _search_collections_node(self, state: AgentState) -> AgentState:
        """Node for CMR collections search."""
        try:
            query_context = state["query_context"]
            collections = await self.cmr_api_agent.search_collections(query_context)
            
            state["cmr_results"]["collections"] = collections
            state["agent_responses"].append(AgentResponse(
                agent_name="cmr_collections_search",
                status="success",
                data={"collection_count": len(collections)}
            ))
            
        except Exception as e:
            state["errors"].append(f"Collections search failed: {str(e)}")
            state["agent_responses"].append(AgentResponse(
                agent_name="cmr_collections_search",
                status="error",
                error=str(e)
            ))
        
        return state
    
    async def _search_granules_node(self, state: AgentState) -> AgentState:
        """Node for CMR granules search."""
        try:
            query_context = state["query_context"]
            collections = state["cmr_results"].get("collections", [])
            
            # Search granules for each collection (parallel execution)
            granule_tasks = [
                self.cmr_api_agent.search_granules(query_context, collection.concept_id)
                for collection in collections[:5]  # Limit to top 5 collections
            ]
            
            granule_results = await asyncio.gather(*granule_tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            all_granules = []
            for i, result in enumerate(granule_results):
                if isinstance(result, Exception):
                    state["errors"].append(f"Granule search failed for collection {i}: {str(result)}")
                else:
                    all_granules.extend(result)
            
            state["cmr_results"]["granules"] = all_granules
            state["agent_responses"].append(AgentResponse(
                agent_name="cmr_granules_search",
                status="success",
                data={"granule_count": len(all_granules)}
            ))
            
        except Exception as e:
            state["errors"].append(f"Granules search failed: {str(e)}")
            state["agent_responses"].append(AgentResponse(
                agent_name="cmr_granules_search",
                status="error",
                error=str(e)
            ))
        
        return state
    
    async def _analyze_data_node(self, state: AgentState) -> AgentState:
        """Node for data analysis and recommendation generation."""
        try:
            query_context = state["query_context"]
            cmr_results = state["cmr_results"]
            sub_queries = state.get("sub_queries")
            
            # Perform analysis on the retrieved data (enhanced or basic)
            if hasattr(self.analysis_agent, 'analyze_results_enhanced'):
                analysis_results = await self.analysis_agent.analyze_results_enhanced(
                    query_context, cmr_results
                )
            else:
                analysis_results = await self.analysis_agent.analyze_results(
                    query_context, cmr_results, sub_queries
                )
            
            state["analysis_results"] = analysis_results
            state["agent_responses"].append(AgentResponse(
                agent_name="data_analysis",
                status="success",
                data={
                    "analysis_count": len(analysis_results),
                    "comparative_analysis_included": any(
                        ar.analysis_type == "comparative_analysis" for ar in analysis_results
                    )
                }
            ))
            
        except Exception as e:
            state["errors"].append(f"Data analysis failed: {str(e)}")
            state["agent_responses"].append(AgentResponse(
                agent_name="data_analysis",
                status="error",
                error=str(e)
            ))
        
        return state
    
    async def _synthesize_response_node(self, state: AgentState) -> AgentState:
        """Node for response synthesis and formatting."""
        try:
            query_context = state.get("query_context")
            cmr_results = state.get("cmr_results", {})
            analysis_results = state.get("analysis_results", [])
            errors = state.get("errors", [])
            
            # Synthesize final response
            system_response = await self.response_agent.synthesize_response(
                query_context, cmr_results, analysis_results, errors
            )
            
            state["final_response"] = system_response
            state["agent_responses"].append(AgentResponse(
                agent_name="response_synthesis",
                status="success",
                data={"response_generated": True}
            ))
            
        except Exception as e:
            state["errors"].append(f"Response synthesis failed: {str(e)}")
            state["agent_responses"].append(AgentResponse(
                agent_name="response_synthesis",
                status="error",
                error=str(e)
            ))
        
        return state
    
    def _should_continue_after_validation(self, state: AgentState) -> str:
        """Conditional edge function to determine next step after validation."""
        return state.get("next_agent", "continue")
    
    async def process_query(self, user_query: str) -> SystemResponse:
        """
        Process a user query through the complete multi-agent workflow with supervisor validation.
        
        Args:
            user_query: Natural language query from user
            
        Returns:
            SystemResponse with comprehensive results and analysis
        """
        retry_count = 0
        best_response = None
        
        while retry_count <= self.max_retries:
            # Initialize state with user query
            initial_state = AgentState({
                "messages": [HumanMessage(content=user_query)],
                "original_query": user_query,
                "retry_count": retry_count,
                "query_context": None,
                "cmr_results": {},
                "analysis_results": [],
                "agent_responses": [],
                "errors": [],
                "next_agent": None
            })
            
            # Execute the workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            # Extract response
            if "final_response" in final_state:
                response = final_state["final_response"]
                
                # Validate with supervisor
                validation = await self.supervisor.validate_final_response(
                    user_query, response
                )
                
                # Integrate supervisor feedback into learning system
                await supervisor_learning.process_supervisor_feedback(
                    user_query, final_state.get("query_context"), validation, "workflow"
                )
                
                # Learn from this interaction
                adaptive_learner.learn_from_query(
                    query=user_query,
                    context=final_state.get("query_context"),
                    response=response,
                    execution_time=response.total_execution_time_ms,
                    success=validation.passed
                )
                
                # Store best response
                if best_response is None or validation.quality_score.overall > best_response[1]:
                    best_response = (response, validation.quality_score.overall)
                
                # Check if validation passed
                if validation.passed:
                    # Add validation metadata to response
                    response.metadata = response.metadata or {}
                    response.metadata["quality_score"] = validation.quality_score.dict()
                    response.metadata["supervisor_validated"] = True
                    return response
                
                # If validation failed and retry is needed
                if validation.requires_retry and retry_count < self.max_retries:
                    logger.info(f"Supervisor requested retry: {validation.retry_guidance}")
                    retry_count += 1
                    
                    # Add retry guidance to state for next iteration
                    initial_state["retry_guidance"] = validation.retry_guidance
                    initial_state["validation_issues"] = validation.issues
                    continue
                else:
                    # Return best response even if not perfect
                    if best_response:
                        response = best_response[0]
                        response.warnings = response.warnings or []
                        response.warnings.append(
                            f"Quality validation score: {best_response[1]:.2f}. "
                            f"Some aspects may need improvement."
                        )
                        return response
            
            # If no valid response, create error response
            if retry_count == self.max_retries:
                return SystemResponse(
                    query_id="error",
                    original_query=user_query,
                    intent="exploratory",
                    recommendations=[],
                    summary="Query processing failed after multiple attempts.",
                    execution_plan=["error_handling"],
                    total_execution_time_ms=0,
                    agent_responses=final_state.get("agent_responses", []),
                    success=False,
                    warnings=final_state.get("errors", [])
                )
            
            retry_count += 1
        
        # Fallback error response
        return SystemResponse(
            query_id="error",
            original_query=user_query,
            intent="exploratory",
            recommendations=[],
            summary="Query processing failed due to errors in the workflow.",
            execution_plan=["error_handling"],
            total_execution_time_ms=0,
            agent_responses=[],
            success=False,
            warnings=["Maximum retries exceeded"]
        )
    
    async def cleanup(self):
        """Cleanup resources after processing."""
        # Cleanup enhanced analysis agent resources if needed
        if hasattr(self.analysis_agent, 'close'):
            try:
                await self.analysis_agent.close()
            except Exception as cleanup_error:
                logger.warning(f"Cleanup warning: {cleanup_error}")
        
        # Close scratchpads
        await self.scratchpad_manager.close_all()