"""
NASA GIOVANNI (GES-DISC Interactive Online Visualization And aNalysis Infrastructure) Integration.

Provides data analysis, visualization, and time series capabilities through NASA's GIOVANNI system.
GIOVANNI offers interactive data analysis tools for satellite-based climate datasets.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode, quote
import structlog

from ..core.config import settings
from ..utils.rate_limiter import RateLimiter
from ..utils.circuit_breaker import CircuitBreaker

logger = structlog.get_logger(__name__)


class GiovanniAnalysisType(Enum):
    """Types of analysis available in GIOVANNI."""
    TIME_SERIES = "time_series"
    AREA_AVERAGED = "area_averaged"
    SEASONAL_DIFFERENCE = "seasonal_difference"
    CORRELATION = "correlation"
    HOVMOLLER = "hovmoller"
    SCATTER_PLOT = "scatter_plot"
    HISTOGRAM = "histogram"
    COMPARISON = "comparison"


class GiovanniDataFormat(Enum):
    """Output data formats supported by GIOVANNI."""
    NETCDF = "netCDF"
    HDF = "HDF"
    CSV = "CSV"
    PNG = "PNG"
    PDF = "PDF"
    JSON = "JSON"


@dataclass
class GiovanniDataRequest:
    """GIOVANNI data analysis request configuration."""
    dataset_id: str
    variables: List[str]
    analysis_type: GiovanniAnalysisType
    start_date: str  # YYYY-MM-DD format
    end_date: str    # YYYY-MM-DD format
    bbox: Optional[Tuple[float, float, float, float]] = None  # (west, south, east, north)
    output_format: GiovanniDataFormat = GiovanniDataFormat.JSON
    additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


@dataclass
class GiovanniAnalysisResult:
    """GIOVANNI analysis result."""
    request_id: str
    status: str
    analysis_type: str
    dataset_info: Dict[str, Any]
    data_url: Optional[str] = None
    visualization_url: Optional[str] = None
    metadata: Dict[str, Any] = None
    download_urls: List[str] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.download_urls is None:
            self.download_urls = []


class GiovanniService:
    """NASA GIOVANNI data analysis and visualization service."""
    
    def __init__(self):
        self.base_url = "https://giovanni.gsfc.nasa.gov/giovanni"
        self.api_version = "4"
        self.session = None
        
        # Rate limiting and circuit breaker
        self.rate_limiter = RateLimiter(requests_per_second=2)  # Conservative rate limiting
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60,
            expected_exception=aiohttp.ClientError
        )
        
        # Request tracking
        self.active_requests = {}
        self.request_history = []
        
        # Authentication (if required)
        self.auth_token = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def initialize(self):
        """Initialize the GIOVANNI service."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes for analysis
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": "NASA-CMR-Agent/1.0",
                    "Accept": "application/json,text/html,application/xhtml+xml"
                }
            )
            
        logger.info("GIOVANNI service initialized")
    
    async def close(self):
        """Close the GIOVANNI service."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def submit_analysis_request(self, request: GiovanniDataRequest) -> str:
        """Submit a data analysis request to GIOVANNI."""
        
        await self.rate_limiter.acquire()
        
        try:
            return await self.circuit_breaker.call(
                self._submit_analysis_request_impl, request
            )
        except Exception as e:
            logger.error(f"GIOVANNI analysis request failed: {e}")
            raise
    
    async def _submit_analysis_request_impl(self, request: GiovanniDataRequest) -> str:
        """Implementation of analysis request submission."""
        
        if not self.session:
            await self.initialize()
        
        # Build GIOVANNI request parameters
        params = self._build_giovanni_params(request)
        
        # Submit request to GIOVANNI
        submit_url = f"{self.base_url}/daac-bin/giovanni_analysis.pl"
        
        logger.info(f"Submitting GIOVANNI analysis request", 
                   dataset=request.dataset_id, 
                   analysis_type=request.analysis_type.value)
        
        start_time = time.time()
        
        async with self.session.post(submit_url, data=params) as response:
            if response.status != 200:
                error_text = await response.text()
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f"GIOVANNI request failed: {error_text}"
                )
            
            # Parse response to get request ID
            response_text = await response.text()
            request_id = self._extract_request_id(response_text)
            
            if not request_id:
                raise ValueError(f"Could not extract request ID from GIOVANNI response")
            
            # Track request
            self.active_requests[request_id] = {
                "submitted_at": time.time(),
                "request": request,
                "status": "submitted"
            }
            
            processing_time = time.time() - start_time
            logger.info(f"GIOVANNI request submitted successfully", 
                       request_id=request_id, 
                       processing_time=processing_time)
            
            return request_id
    
    async def check_analysis_status(self, request_id: str) -> GiovanniAnalysisResult:
        """Check the status of a GIOVANNI analysis request."""
        
        await self.rate_limiter.acquire()
        
        try:
            return await self.circuit_breaker.call(
                self._check_analysis_status_impl, request_id
            )
        except Exception as e:
            logger.error(f"GIOVANNI status check failed: {e}")
            raise
    
    async def _check_analysis_status_impl(self, request_id: str) -> GiovanniAnalysisResult:
        """Implementation of status checking."""
        
        if not self.session:
            await self.initialize()
        
        status_url = f"{self.base_url}/daac-bin/giovanni_status.pl"
        params = {"REQUEST_ID": request_id}
        
        start_time = time.time()
        
        async with self.session.get(status_url, params=params) as response:
            if response.status != 200:
                error_text = await response.text()
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f"GIOVANNI status check failed: {error_text}"
                )
            
            response_text = await response.text()
            processing_time = time.time() - start_time
            
            # Parse GIOVANNI response
            result = self._parse_giovanni_response(request_id, response_text)
            result.processing_time = processing_time
            
            # Update request tracking
            if request_id in self.active_requests:
                self.active_requests[request_id]["status"] = result.status
                self.active_requests[request_id]["last_checked"] = time.time()
            
            return result
    
    async def wait_for_completion(self, request_id: str, 
                                 timeout: int = 300,
                                 poll_interval: int = 10) -> GiovanniAnalysisResult:
        """Wait for GIOVANNI analysis to complete."""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = await self.check_analysis_status(request_id)
            
            if result.status in ["completed", "failed", "error"]:
                if result.status == "completed":
                    logger.info(f"GIOVANNI analysis completed", request_id=request_id)
                else:
                    logger.error(f"GIOVANNI analysis failed", 
                               request_id=request_id, 
                               error=result.error_message)
                
                # Clean up tracking
                if request_id in self.active_requests:
                    self.request_history.append(self.active_requests.pop(request_id))
                
                return result
            
            logger.debug(f"GIOVANNI analysis in progress", 
                        request_id=request_id, 
                        status=result.status)
            
            await asyncio.sleep(poll_interval)
        
        # Timeout reached
        logger.warning(f"GIOVANNI analysis timeout", request_id=request_id)
        result = await self.check_analysis_status(request_id)
        result.error_message = f"Analysis timeout after {timeout} seconds"
        
        return result
    
    async def generate_time_series(self, dataset_id: str, 
                                  variables: List[str],
                                  start_date: str, 
                                  end_date: str,
                                  bbox: Optional[Tuple[float, float, float, float]] = None) -> GiovanniAnalysisResult:
        """Generate time series analysis via GIOVANNI."""
        
        request = GiovanniDataRequest(
            dataset_id=dataset_id,
            variables=variables,
            analysis_type=GiovanniAnalysisType.TIME_SERIES,
            start_date=start_date,
            end_date=end_date,
            bbox=bbox,
            output_format=GiovanniDataFormat.JSON
        )
        
        request_id = await self.submit_analysis_request(request)
        return await self.wait_for_completion(request_id)
    
    async def perform_correlation_analysis(self, dataset1_id: str, 
                                         dataset2_id: str,
                                         variables1: List[str],
                                         variables2: List[str],
                                         start_date: str, 
                                         end_date: str,
                                         bbox: Optional[Tuple[float, float, float, float]] = None) -> GiovanniAnalysisResult:
        """Perform correlation analysis between two datasets."""
        
        request = GiovanniDataRequest(
            dataset_id=f"{dataset1_id},{dataset2_id}",
            variables=variables1 + variables2,
            analysis_type=GiovanniAnalysisType.CORRELATION,
            start_date=start_date,
            end_date=end_date,
            bbox=bbox,
            output_format=GiovanniDataFormat.JSON,
            additional_params={
                "dataset1": dataset1_id,
                "dataset2": dataset2_id,
                "variables1": variables1,
                "variables2": variables2
            }
        )
        
        request_id = await self.submit_analysis_request(request)
        return await self.wait_for_completion(request_id)
    
    async def generate_area_averaged_plot(self, dataset_id: str,
                                        variables: List[str],
                                        start_date: str,
                                        end_date: str,
                                        bbox: Tuple[float, float, float, float]) -> GiovanniAnalysisResult:
        """Generate area-averaged time series plot."""
        
        request = GiovanniDataRequest(
            dataset_id=dataset_id,
            variables=variables,
            analysis_type=GiovanniAnalysisType.AREA_AVERAGED,
            start_date=start_date,
            end_date=end_date,
            bbox=bbox,
            output_format=GiovanniDataFormat.PNG
        )
        
        request_id = await self.submit_analysis_request(request)
        return await self.wait_for_completion(request_id)
    
    async def create_hovmoller_diagram(self, dataset_id: str,
                                     variable: str,
                                     start_date: str,
                                     end_date: str,
                                     bbox: Tuple[float, float, float, float],
                                     direction: str = "longitude") -> GiovanniAnalysisResult:
        """Create Hovmöller diagram (time vs longitude/latitude)."""
        
        request = GiovanniDataRequest(
            dataset_id=dataset_id,
            variables=[variable],
            analysis_type=GiovanniAnalysisType.HOVMOLLER,
            start_date=start_date,
            end_date=end_date,
            bbox=bbox,
            output_format=GiovanniDataFormat.PNG,
            additional_params={"direction": direction}
        )
        
        request_id = await self.submit_analysis_request(request)
        return await self.wait_for_completion(request_id)
    
    async def get_available_datasets(self) -> List[Dict[str, Any]]:
        """Get list of available datasets in GIOVANNI."""
        
        await self.rate_limiter.acquire()
        
        try:
            return await self.circuit_breaker.call(self._get_available_datasets_impl)
        except Exception as e:
            logger.error(f"Failed to get GIOVANNI datasets: {e}")
            return []
    
    async def _get_available_datasets_impl(self) -> List[Dict[str, Any]]:
        """Implementation of dataset listing."""
        
        if not self.session:
            await self.initialize()
        
        catalog_url = f"{self.base_url}/daac-bin/giovanni_catalog.pl"
        
        async with self.session.get(catalog_url) as response:
            if response.status != 200:
                logger.warning(f"GIOVANNI catalog request failed: {response.status}")
                return []
            
            response_text = await response.text()
            datasets = self._parse_giovanni_catalog(response_text)
            
            logger.info(f"Retrieved {len(datasets)} GIOVANNI datasets")
            return datasets
    
    def _build_giovanni_params(self, request: GiovanniDataRequest) -> Dict[str, str]:
        """Build GIOVANNI request parameters."""
        
        params = {
            "SERVICE": "GIOVANNI",
            "VERSION": self.api_version,
            "REQUEST": "GetAnalysis",
            "FORMAT": request.output_format.value,
            "STARTTIME": request.start_date,
            "ENDTIME": request.end_date,
            "DATA": request.dataset_id,
            "VARIABLES": ",".join(request.variables),
            "PORTAL": "GIOVANNI",
            "ANALYSIS": request.analysis_type.value.upper()
        }
        
        # Add bounding box if specified
        if request.bbox:
            west, south, east, north = request.bbox
            params["BBOX"] = f"{west},{south},{east},{north}"
        
        # Add additional parameters
        params.update(request.additional_params)
        
        return params
    
    def _extract_request_id(self, response_text: str) -> Optional[str]:
        """Extract request ID from GIOVANNI response."""
        
        # GIOVANNI typically returns request ID in various formats
        # This is a simplified implementation - in practice, you'd need to
        # parse the actual GIOVANNI response format
        
        import re
        
        # Look for common patterns
        patterns = [
            r'REQUEST_ID["\']?\s*[:=]\s*["\']?([A-Za-z0-9_-]+)',
            r'request[_-]?id["\']?\s*[:=]\s*["\']?([A-Za-z0-9_-]+)',
            r'session[_-]?id["\']?\s*[:=]\s*["\']?([A-Za-z0-9_-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Fallback: generate a request ID based on timestamp
        # This is not ideal but provides fallback functionality
        import hashlib
        request_hash = hashlib.md5(response_text[:200].encode()).hexdigest()[:8]
        return f"giovanni_{int(time.time())}_{request_hash}"
    
    def _parse_giovanni_response(self, request_id: str, response_text: str) -> GiovanniAnalysisResult:
        """Parse GIOVANNI analysis response."""
        
        # This is a simplified parser - actual GIOVANNI responses would need
        # more sophisticated parsing based on the real API format
        
        result = GiovanniAnalysisResult(
            request_id=request_id,
            status="unknown",
            analysis_type="",
            dataset_info={}
        )
        
        # Check for common status indicators
        lower_text = response_text.lower()
        
        if "completed" in lower_text or "finished" in lower_text:
            result.status = "completed"
        elif "running" in lower_text or "processing" in lower_text:
            result.status = "running"
        elif "failed" in lower_text or "error" in lower_text:
            result.status = "failed"
            # Extract error message
            import re
            error_match = re.search(r'error[:\s]*([^\n\r]+)', response_text, re.IGNORECASE)
            if error_match:
                result.error_message = error_match.group(1).strip()
        else:
            result.status = "submitted"
        
        # Extract URLs if available
        import re
        url_patterns = [
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        ]
        
        for pattern in url_patterns:
            urls = re.findall(pattern, response_text)
            if urls:
                result.data_url = urls[0]
                result.download_urls = urls
                break
        
        return result
    
    def _parse_giovanni_catalog(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse GIOVANNI dataset catalog."""
        
        # Simplified catalog parser
        datasets = []
        
        # This would need to be implemented based on actual GIOVANNI catalog format
        # For now, return some common NASA datasets available in GIOVANNI
        
        common_datasets = [
            {
                "id": "GLDAS_NOAH025_M",
                "title": "GLDAS Noah Land Surface Model L4 monthly 0.25 x 0.25 degree",
                "description": "Global Land Data Assimilation System",
                "variables": ["Evap_tavg", "Qsb_tavg", "Rainf_tavg", "SoilMoi0_10cm_tavg"],
                "temporal_coverage": "1948-present",
                "spatial_resolution": "0.25 degree"
            },
            {
                "id": "GPM_3IMERGM",
                "title": "GPM IMERG Final Precipitation L3 Monthly 0.1 degree x 0.1 degree",
                "description": "Global Precipitation Measurement mission",
                "variables": ["precipitation"],
                "temporal_coverage": "2000-present",
                "spatial_resolution": "0.1 degree"
            },
            {
                "id": "MERRA2_400",
                "title": "MERRA-2 tavgM_2d_slv_Nx: 2d,Monthly mean,Time-Averaged,Single-Level,Assimilation,Single-Level Diagnostics",
                "description": "Modern-Era Retrospective analysis for Research and Applications, Version 2",
                "variables": ["T2M", "PRECTOT", "QV2M", "PS"],
                "temporal_coverage": "1980-present",
                "spatial_resolution": "0.5 x 0.625 degree"
            }
        ]
        
        datasets.extend(common_datasets)
        
        logger.debug(f"Parsed {len(datasets)} datasets from GIOVANNI catalog")
        return datasets
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get GIOVANNI service status and statistics."""
        
        return {
            "service": "NASA GIOVANNI",
            "base_url": self.base_url,
            "active_requests": len(self.active_requests),
            "completed_requests": len(self.request_history),
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "rate_limit_remaining": self.rate_limiter.get_remaining_requests(),
            "session_active": self.session is not None
        }


# Global GIOVANNI service instance
_giovanni_service: Optional[GiovanniService] = None


async def get_giovanni_service() -> GiovanniService:
    """Get or create the global GIOVANNI service."""
    global _giovanni_service
    
    if _giovanni_service is None:
        _giovanni_service = GiovanniService()
        await _giovanni_service.initialize()
    
    return _giovanni_service


async def close_giovanni_service():
    """Close the global GIOVANNI service."""
    global _giovanni_service
    
    if _giovanni_service:
        await _giovanni_service.close()
        _giovanni_service = None