import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog
from contextlib import asynccontextmanager

from ..models.schemas import (
    QueryContext, CMRCollection, CMRGranule, 
    SpatialConstraint, TemporalConstraint
)
from ..core.config import settings
from ..services.circuit_breaker import CircuitBreakerService
from ..services.performance_optimizer import cached, measure_time, query_optimizer


logger = structlog.get_logger(__name__)


class CMRAPIAgent:
    """
    NASA CMR API interaction agent with advanced error handling.
    
    Features:
    - Parallel API requests with rate limiting
    - Circuit breaker pattern for fault tolerance
    - Intelligent retry logic with exponential backoff
    - Response caching and metadata enrichment
    - Comprehensive error handling and recovery
    """
    
    def __init__(self):
        self.base_url = settings.cmr_base_url
        self.timeout = settings.cmr_request_timeout
        self.max_retries = settings.cmr_max_retries
        self.rate_limit = settings.cmr_rate_limit_per_second
        
        self.circuit_breaker = CircuitBreakerService(
            failure_threshold=settings.circuit_breaker_failure_threshold,
            recovery_timeout=settings.circuit_breaker_recovery_timeout
        )
        
        # Rate limiting semaphore
        self._rate_limiter = asyncio.Semaphore(self.rate_limit)
        self._last_request_time = datetime.now()
        
        # HTTP client with connection pooling
        self._client = None
    
    @asynccontextmanager
    async def _get_client(self):
        """Get HTTP client with connection pooling."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=10)
            )
        
        try:
            yield self._client
        finally:
            pass  # Keep client open for reuse
    
    async def close(self):
        """Close HTTP client connections."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    @cached(ttl=3600, key_prefix="cmr_collections")
    @measure_time("cmr_collections_search")
    async def search_collections(self, query_context: QueryContext) -> List[CMRCollection]:
        """
        Search NASA CMR for relevant collections based on query context.
        
        Args:
            query_context: Parsed query with constraints and intent
            
        Returns:
            List of relevant CMR collections
        """
        async with self._rate_limiter:
            await self._enforce_rate_limit()
            
            try:
                params = self._build_collection_search_params(query_context)
                
                async with await self.circuit_breaker.call():
                    collections_data = await self._make_cmr_request("/collections", params)
                
                collections = []
                for item in collections_data.get("feed", {}).get("entry", []):
                    try:
                        collection = self._parse_collection(item)
                        collections.append(collection)
                    except Exception as e:
                        logger.warning("Failed to parse collection", error=str(e), item_id=item.get("id"))
                        continue
                
                logger.info("Collections search completed", count=len(collections))
                return collections
                
            except Exception as e:
                logger.error("Collections search failed", error=str(e))
                raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    @cached(ttl=1800, key_prefix="cmr_granules")
    @measure_time("cmr_granules_search")
    async def search_granules(
        self, 
        query_context: QueryContext, 
        collection_concept_id: str,
        limit: int = 100
    ) -> List[CMRGranule]:
        """
        Search granules for a specific collection.
        
        Args:
            query_context: Parsed query with constraints
            collection_concept_id: CMR collection ID
            limit: Maximum granules to retrieve
            
        Returns:
            List of granules for the collection
        """
        async with self._rate_limiter:
            await self._enforce_rate_limit()
            
            try:
                params = self._build_granule_search_params(query_context, collection_concept_id, limit)
                
                async with await self.circuit_breaker.call():
                    granules_data = await self._make_cmr_request("/granules", params)
                
                granules = []
                for item in granules_data.get("feed", {}).get("entry", []):
                    try:
                        granule = self._parse_granule(item)
                        granules.append(granule)
                    except Exception as e:
                        logger.warning("Failed to parse granule", error=str(e), item_id=item.get("id"))
                        continue
                
                logger.info("Granules search completed", 
                           collection_id=collection_concept_id, count=len(granules))
                return granules
                
            except Exception as e:
                logger.error("Granules search failed", 
                            collection_id=collection_concept_id, error=str(e))
                raise
    
    async def get_collection_variables(self, collection_concept_id: str) -> List[Dict[str, Any]]:
        """Get variables associated with a collection."""
        async with self._rate_limiter:
            await self._enforce_rate_limit()
            
            try:
                params = {"collection_concept_id": collection_concept_id}
                
                async with await self.circuit_breaker.call():
                    variables_data = await self._make_cmr_request("/variables", params)
                
                variables = variables_data.get("feed", {}).get("entry", [])
                logger.info("Variables retrieved", 
                           collection_id=collection_concept_id, count=len(variables))
                return variables
                
            except Exception as e:
                logger.error("Variables retrieval failed", 
                            collection_id=collection_concept_id, error=str(e))
                return []
    
    def _build_collection_search_params(self, query_context: QueryContext) -> Dict[str, Any]:
        """Build CMR API parameters for collection search."""
        params = {
            "page_size": 50,
            "pretty": True,
            "sort_key": "-usage_score"  # Sort by relevance/usage
        }
        
        constraints = query_context.constraints
        
        # Spatial constraints
        if constraints.spatial:
            if constraints.spatial.region_name:
                # For named regions, we'd need a mapping service
                # For now, use bounding box if available
                pass
            
            if all([constraints.spatial.north, constraints.spatial.south,
                   constraints.spatial.east, constraints.spatial.west]):
                bbox = f"{constraints.spatial.west},{constraints.spatial.south}," \
                       f"{constraints.spatial.east},{constraints.spatial.north}"
                params["bounding_box"] = bbox
        
        # Temporal constraints
        if constraints.temporal:
            if constraints.temporal.start_date and constraints.temporal.end_date:
                start = constraints.temporal.start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                end = constraints.temporal.end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                params["temporal"] = f"{start},{end}"
        
        # Science keywords
        if constraints.keywords:
            # CMR uses science_keywords parameter
            for i, keyword in enumerate(constraints.keywords[:5]):  # Limit to 5
                params[f"science_keywords[{i}][term]"] = keyword
        
        # Platforms
        if constraints.platforms:
            for i, platform in enumerate(constraints.platforms[:3]):
                params[f"platform[{i}]"] = platform
        
        # Instruments
        if constraints.instruments:
            for i, instrument in enumerate(constraints.instruments[:3]):
                params[f"instrument[{i}]"] = instrument
        
        # Data type constraints
        if constraints.data_types:
            # This would need mapping to CMR processing levels or other filters
            pass
        
        return params
    
    def _build_granule_search_params(
        self, 
        query_context: QueryContext, 
        collection_concept_id: str,
        limit: int
    ) -> Dict[str, Any]:
        """Build CMR API parameters for granule search."""
        params = {
            "collection_concept_id": collection_concept_id,
            "page_size": min(limit, 100),
            "sort_key": "-start_date"  # Most recent first
        }
        
        constraints = query_context.constraints
        
        # Spatial constraints (same as collections)
        if constraints.spatial and all([
            constraints.spatial.north, constraints.spatial.south,
            constraints.spatial.east, constraints.spatial.west
        ]):
            bbox = f"{constraints.spatial.west},{constraints.spatial.south}," \
                   f"{constraints.spatial.east},{constraints.spatial.north}"
            params["bounding_box"] = bbox
        
        # Temporal constraints
        if constraints.temporal:
            if constraints.temporal.start_date and constraints.temporal.end_date:
                start = constraints.temporal.start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                end = constraints.temporal.end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                params["temporal"] = f"{start},{end}"
        
        return params
    
    async def _make_cmr_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make request to CMR API with error handling."""
        url = f"{self.base_url}{endpoint}.json"
        
        async with self._get_client() as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                logger.debug("CMR request successful", 
                           endpoint=endpoint, status=response.status_code)
                return data
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    logger.warning("Rate limited by CMR API")
                    await asyncio.sleep(2)  # Wait before retry
                raise
            except httpx.RequestError as e:
                logger.error("CMR request failed", endpoint=endpoint, error=str(e))
                raise
    
    def _parse_collection(self, cmr_data: Dict[str, Any]) -> CMRCollection:
        """Parse CMR collection data into structured format."""
        # Extract basic metadata
        concept_id = cmr_data.get("id", "")
        title = cmr_data.get("title", "")
        summary = cmr_data.get("summary", "")
        
        # Handle different data structures
        short_name = ""
        version_id = ""
        data_center = ""
        
        if "short_name" in cmr_data:
            short_name = cmr_data["short_name"]
        if "version_id" in cmr_data:
            version_id = cmr_data["version_id"]
        if "data_center" in cmr_data:
            data_center = cmr_data["data_center"]
        elif "organizations" in cmr_data and cmr_data["organizations"]:
            data_center = cmr_data["organizations"][0]
        
        # Extract platforms and instruments
        platforms = []
        instruments = []
        
        if "platforms" in cmr_data:
            platforms = [p.get("short_name", "") for p in cmr_data["platforms"]]
            for platform in cmr_data["platforms"]:
                if "instruments" in platform:
                    instruments.extend([i.get("short_name", "") for i in platform["instruments"]])
        
        # Extract temporal coverage
        temporal_coverage = None
        if "time_start" in cmr_data or "time_end" in cmr_data:
            temporal_coverage = {
                "start": cmr_data.get("time_start"),
                "end": cmr_data.get("time_end")
            }
        
        # Extract spatial coverage
        spatial_coverage = None
        if "boxes" in cmr_data and cmr_data["boxes"]:
            # CMR boxes format: [south, west, north, east]
            box = cmr_data["boxes"][0].split()
            if len(box) == 4:
                spatial_coverage = {
                    "south": float(box[0]),
                    "west": float(box[1]),
                    "north": float(box[2]),
                    "east": float(box[3])
                }
        
        # Extract flags
        cloud_hosted = cmr_data.get("cloud_hosted", False)
        online_access_flag = cmr_data.get("online_access_flag", False)
        
        return CMRCollection(
            concept_id=concept_id,
            short_name=short_name,
            version_id=version_id,
            title=title,
            summary=summary,
            data_center=data_center,
            platforms=platforms,
            instruments=instruments,
            temporal_coverage=temporal_coverage,
            spatial_coverage=spatial_coverage,
            variables=[],  # Would need separate API call
            processing_level=cmr_data.get("processing_level_id"),
            cloud_hosted=cloud_hosted,
            online_access_flag=online_access_flag
        )
    
    def _parse_granule(self, cmr_data: Dict[str, Any]) -> CMRGranule:
        """Parse CMR granule data into structured format."""
        concept_id = cmr_data.get("id", "")
        title = cmr_data.get("title", "")
        collection_concept_id = cmr_data.get("collection_concept_id", "")
        producer_granule_id = cmr_data.get("producer_granule_id", title)
        
        # Extract temporal extent
        temporal_extent = None
        if "time_start" in cmr_data or "time_end" in cmr_data:
            temporal_extent = {
                "start": cmr_data.get("time_start"),
                "end": cmr_data.get("time_end")
            }
        
        # Extract spatial extent
        spatial_extent = None
        if "boxes" in cmr_data and cmr_data["boxes"]:
            box = cmr_data["boxes"][0].split()
            if len(box) == 4:
                spatial_extent = {
                    "south": float(box[0]),
                    "west": float(box[1]),
                    "north": float(box[2]),
                    "east": float(box[3])
                }
        
        # Extract size if available
        size_mb = None
        if "granule_size" in cmr_data:
            try:
                size_mb = float(cmr_data["granule_size"])
            except (ValueError, TypeError):
                pass
        
        # Extract links
        links = cmr_data.get("links", [])
        
        return CMRGranule(
            concept_id=concept_id,
            title=title,
            collection_concept_id=collection_concept_id,
            producer_granule_id=producer_granule_id,
            temporal_extent=temporal_extent,
            spatial_extent=spatial_extent,
            size_mb=size_mb,
            cloud_cover=cmr_data.get("cloud_cover"),
            day_night_flag=cmr_data.get("day_night_flag"),
            links=links
        )
    
    async def _enforce_rate_limit(self):
        """Enforce rate limiting between requests."""
        now = datetime.now()
        time_since_last = (now - self._last_request_time).total_seconds()
        
        min_interval = 1.0 / self.rate_limit
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self._last_request_time = datetime.now()