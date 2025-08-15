"""
NASA MODAPS/LAADS DAAC Integration Service.

Provides access to NASA's MODIS Adaptive Processing System (MODAPS) and
Land, Atmosphere Near real-time Capability for EOS (LANCE) data through
the Level-1 and Atmosphere Archive & Distribution System (LAADS) DAAC.

This service enables discovery and access to atmospheric and land surface
data products from Terra and Aqua satellites.
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
from .earthdata_auth import get_earthdata_auth_service

logger = structlog.get_logger(__name__)


class MODISProduct(Enum):
    """MODIS data products available through LAADS DAAC."""
    # Atmosphere Products
    MOD04_L2 = "MOD04_L2"  # Aerosol 5-Min L2 Swath 10km
    MYD04_L2 = "MYD04_L2"  # Aerosol 5-Min L2 Swath 10km (Aqua)
    MOD06_L2 = "MOD06_L2"  # Clouds 5-Min L2 Swath 1km and 5km
    MOD07_L2 = "MOD07_L2"  # Atmospheric Profiles 5-Min L2 Swath 5km
    MOD08_D3 = "MOD08_D3"  # Aerosol Daily L3 Global 1Deg CMG
    MOD08_M3 = "MOD08_M3"  # Aerosol Monthly L3 Global 1Deg CMG
    
    # Land Products
    MOD09A1 = "MOD09A1"    # Surface Reflectance 8-Day L3 Global 500m SIN Grid
    MOD11A1 = "MOD11A1"    # Land Surface Temperature/Emissivity Daily L3 Global 1km SIN Grid
    MOD13Q1 = "MOD13Q1"    # Vegetation Indices 16-Day L3 Global 250m SIN Grid
    MOD15A2H = "MOD15A2H"  # Leaf Area Index/FPAR 8-Day L4 Global 500m SIN Grid
    MOD17A2H = "MOD17A2H"  # Gross Primary Productivity 8-Day L4 Global 500m SIN Grid
    
    # Fire Products
    MOD14A1 = "MOD14A1"    # Thermal Anomalies/Fire Daily L3 Global 1km SIN Grid


class LADDSDataFormat(Enum):
    """Data formats available from LAADS DAAC."""
    HDF4 = "hdf"
    HDF5 = "h5"
    NETCDF = "nc"
    GEOTIFF = "tif"
    JPEG = "jpg"
    JSON = "json"


@dataclass
class MODAPSDataRequest:
    """MODAPS/LAADS data request configuration."""
    product: MODISProduct
    collection: str  # Collection version (e.g., "61", "6.1")
    start_date: str  # YYYY-MM-DD format
    end_date: str    # YYYY-MM-DD format
    bbox: Optional[Tuple[float, float, float, float]] = None  # (west, south, east, north)
    day_night_flag: Optional[str] = None  # "D", "N", or "B" (both)
    data_format: LADDSDataFormat = LADDSDataFormat.HDF4
    download_links_only: bool = True
    
    def __post_init__(self):
        if isinstance(self.product, str):
            self.product = MODISProduct(self.product)
        if isinstance(self.data_format, str):
            self.data_format = LADDSDataFormat(self.data_format)


@dataclass 
class MODISDataFile:
    """MODIS data file information."""
    file_id: str
    file_name: str
    product: str
    collection: str
    processing_date: str
    file_size: int  # bytes
    checksum: str
    download_url: str
    browse_url: Optional[str] = None
    metadata_url: Optional[str] = None
    temporal_coverage: Optional[Dict[str, str]] = None
    spatial_coverage: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.temporal_coverage is None:
            self.temporal_coverage = {}
        if self.spatial_coverage is None:
            self.spatial_coverage = {}


@dataclass
class MODAPSSearchResult:
    """MODAPS search result."""
    request_id: str
    total_files: int
    files: List[MODISDataFile]
    search_params: Dict[str, Any]
    processing_time: Optional[float] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class MODAPSLAADSService:
    """NASA MODAPS/LAADS DAAC data access service."""
    
    def __init__(self):
        # LAADS Web Service endpoints
        self.base_url = "https://ladsweb.modaps.eosdis.nasa.gov"
        self.api_url = f"{self.base_url}/api/v2"
        self.search_url = f"{self.api_url}/content/details"
        self.download_url = f"{self.api_url}/content/archives"
        
        # Service configuration
        self.session = None
        self.auth_service = None
        
        # Rate limiting and circuit breaker
        self.rate_limiter = RateLimiter(requests_per_second=3)  # Conservative for LAADS
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=120,  # 2 minutes recovery
            expected_exception=aiohttp.ClientError
        )
        
        # Request tracking
        self.active_requests = {}
        self.request_history = []
        
        # API key (from Earthdata Login)
        self.api_key = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def initialize(self):
        """Initialize the MODAPS/LAADS service."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=180)  # 3 minutes for large searches
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": "NASA-CMR-Agent/1.0",
                    "Accept": "application/json"
                }
            )
        
        # Initialize Earthdata authentication
        self.api_key = getattr(settings, 'laads_api_key', None)
        try:
            self.auth_service = await get_earthdata_auth_service()
            # Get API key from authenticated session
            if self.auth_service and self.auth_service.is_authenticated():
                if not self.api_key and self.auth_service.current_token:
                    # Use Earthdata token as API key
                    self.api_key = self.auth_service.current_token.access_token
        except Exception as e:
            logger.debug(f"Earthdata authentication not available: {e}")
        
        # Log initialization status
        if self.api_key:
            logger.info("MODAPS/LAADS service initialized with API key")
        else:
            logger.info("MODAPS/LAADS service initialized without API key - some features may be limited")
    
    async def close(self):
        """Close the MODAPS/LAADS service."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def search_modis_data(self, request: MODAPSDataRequest) -> MODAPSSearchResult:
        """Search for MODIS data products."""
        
        await self.rate_limiter.acquire()
        
        try:
            return await self.circuit_breaker.call(
                self._search_modis_data_impl, request
            )
        except Exception as e:
            logger.error(f"MODIS data search failed: {e}")
            raise
    
    async def _search_modis_data_impl(self, request: MODAPSDataRequest) -> MODAPSSearchResult:
        """Implementation of MODIS data search."""
        
        if not self.session:
            await self.initialize()
        
        # Build search parameters
        params = self._build_search_params(request)
        
        logger.info(f"Searching MODIS data", 
                   product=request.product.value,
                   collection=request.collection,
                   date_range=f"{request.start_date} to {request.end_date}")
        
        start_time = time.time()
        request_id = f"modis_{int(start_time)}_{request.product.value}"
        
        # Add authentication if available
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            async with self.session.get(self.search_url, params=params, headers=headers) as response:
                if response.status == 401:
                    raise ValueError("Authentication required for LAADS DAAC access")
                elif response.status == 403:
                    raise ValueError("Access denied - check API key permissions")
                elif response.status != 200:
                    error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"LAADS search failed: {error_text}"
                    )
                
                # Parse response
                response_data = await response.json()
                processing_time = time.time() - start_time
                
                # Process search results
                result = self._parse_search_response(
                    request_id, request, response_data, processing_time
                )
                
                # Track request
                self.active_requests[request_id] = {
                    "submitted_at": start_time,
                    "request": request,
                    "status": "completed",
                    "result": result
                }
                
                logger.info(f"MODIS search completed", 
                           request_id=request_id,
                           files_found=result.total_files,
                           processing_time=processing_time)
                
                return result
                
        except Exception as e:
            logger.error(f"MODIS search request failed: {e}")
            # Return empty result on error
            return MODAPSSearchResult(
                request_id=request_id,
                total_files=0,
                files=[],
                search_params=asdict(request),
                processing_time=time.time() - start_time,
                warnings=[f"Search failed: {str(e)}"]
            )
    
    async def get_product_info(self, product: MODISProduct) -> Dict[str, Any]:
        """Get detailed information about a MODIS product."""
        
        await self.rate_limiter.acquire()
        
        try:
            return await self.circuit_breaker.call(
                self._get_product_info_impl, product
            )
        except Exception as e:
            logger.error(f"Failed to get product info: {e}")
            return {}
    
    async def _get_product_info_impl(self, product: MODISProduct) -> Dict[str, Any]:
        """Implementation of product info retrieval."""
        
        if not self.session:
            await self.initialize()
        
        info_url = f"{self.api_url}/content/products/{product.value}"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        async with self.session.get(info_url, headers=headers) as response:
            if response.status != 200:
                logger.warning(f"Product info request failed: {response.status}")
                # Return basic product information as fallback
                return self._get_fallback_product_info(product)
            
            return await response.json()
    
    async def get_available_collections(self, product: MODISProduct) -> List[str]:
        """Get available collection versions for a product."""
        
        product_info = await self.get_product_info(product)
        
        if "collections" in product_info:
            return [str(c["version"]) for c in product_info["collections"]]
        
        # Fallback collection versions
        return ["61", "6.1", "6", "5"]  # Common MODIS collection versions
    
    async def download_file_list(self, result: MODAPSSearchResult, 
                                output_format: str = "csv") -> str:
        """Generate download file list in specified format."""
        
        if output_format.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "file_name", "product", "collection", "file_size_mb", 
                "download_url", "processing_date", "checksum"
            ])
            
            # Write file entries
            for file_info in result.files:
                writer.writerow([
                    file_info.file_name,
                    file_info.product,
                    file_info.collection,
                    round(file_info.file_size / (1024 * 1024), 2),  # MB
                    file_info.download_url,
                    file_info.processing_date,
                    file_info.checksum
                ])
            
            return output.getvalue()
        
        elif output_format.lower() == "json":
            return json.dumps([asdict(f) for f in result.files], indent=2)
        
        elif output_format.lower() == "urls":
            return "\n".join([f.download_url for f in result.files])
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    async def get_temporal_coverage(self, product: MODISProduct, 
                                  collection: str) -> Dict[str, str]:
        """Get temporal coverage information for a product/collection."""
        
        product_info = await self.get_product_info(product)
        
        # Extract temporal coverage from product metadata
        if "temporal_coverage" in product_info:
            return product_info["temporal_coverage"]
        
        # Fallback temporal coverage based on product type
        temporal_info = {
            "start_date": "2000-02-24",  # Terra launch
            "end_date": "ongoing",
            "temporal_resolution": "daily"
        }
        
        # Adjust for Aqua products
        if product.value.startswith("MYD"):
            temporal_info["start_date"] = "2002-07-04"  # Aqua launch
        
        return temporal_info
    
    def _build_search_params(self, request: MODAPSDataRequest) -> Dict[str, str]:
        """Build LAADS API search parameters."""
        
        params = {
            "products": request.product.value,
            "collection": request.collection,
            "dateRanges": f"{request.start_date},{request.end_date}",
            "areaOfInterest": "global"  # Default to global
        }
        
        # Add spatial bounding box if specified
        if request.bbox:
            west, south, east, north = request.bbox
            params["areaOfInterest"] = f"bbox:{west},{south},{east},{north}"
        
        # Add day/night flag if specified
        if request.day_night_flag:
            params["dayNightFlag"] = request.day_night_flag
        
        # Add data format preference
        params["format"] = request.data_format.value
        
        # Request download links
        if request.download_links_only:
            params["getLinks"] = "true"
        
        return params
    
    def _parse_search_response(self, request_id: str, request: MODAPSDataRequest,
                              response_data: Dict[str, Any], 
                              processing_time: float) -> MODAPSSearchResult:
        """Parse LAADS search response."""
        
        files = []
        warnings = []
        
        # Handle different response formats
        if "content" in response_data:
            file_entries = response_data["content"]
        elif "files" in response_data:
            file_entries = response_data["files"]
        else:
            file_entries = response_data.get("data", [])
        
        # Process file entries
        for entry in file_entries:
            try:
                file_info = self._parse_file_entry(entry, request.product.value)
                files.append(file_info)
            except Exception as e:
                warnings.append(f"Failed to parse file entry: {e}")
                logger.warning(f"File parsing error: {e}")
        
        return MODAPSSearchResult(
            request_id=request_id,
            total_files=len(files),
            files=files,
            search_params=asdict(request),
            processing_time=processing_time,
            warnings=warnings
        )
    
    def _parse_file_entry(self, entry: Dict[str, Any], product: str) -> MODISDataFile:
        """Parse individual file entry from LAADS response."""
        
        # Extract file information
        file_name = entry.get("name", entry.get("fileName", ""))
        file_id = entry.get("id", entry.get("fileId", file_name))
        
        # File size handling
        file_size = entry.get("size", entry.get("fileSize", 0))
        if isinstance(file_size, str):
            try:
                file_size = int(file_size)
            except ValueError:
                file_size = 0
        
        # Download URL construction
        download_url = entry.get("downloadsLink", entry.get("downloadUrl", ""))
        if not download_url and "archive" in entry:
            download_url = f"{self.download_url}/{entry['archive']}"
        
        # Browse/preview URL
        browse_url = entry.get("browseUrl", entry.get("previewUrl"))
        
        # Metadata URL
        metadata_url = entry.get("metadataUrl")
        
        # Temporal coverage
        temporal_coverage = {}
        if "startTime" in entry and "endTime" in entry:
            temporal_coverage = {
                "start_time": entry["startTime"],
                "end_time": entry["endTime"]
            }
        
        # Spatial coverage
        spatial_coverage = {}
        if "bbox" in entry:
            bbox = entry["bbox"]
            if isinstance(bbox, list) and len(bbox) == 4:
                spatial_coverage = {
                    "west": bbox[0],
                    "south": bbox[1], 
                    "east": bbox[2],
                    "north": bbox[3]
                }
        
        return MODISDataFile(
            file_id=file_id,
            file_name=file_name,
            product=product,
            collection=entry.get("collection", ""),
            processing_date=entry.get("processingDate", entry.get("createdAt", "")),
            file_size=file_size,
            checksum=entry.get("checksum", entry.get("hash", "")),
            download_url=download_url,
            browse_url=browse_url,
            metadata_url=metadata_url,
            temporal_coverage=temporal_coverage,
            spatial_coverage=spatial_coverage
        )
    
    def _get_fallback_product_info(self, product: MODISProduct) -> Dict[str, Any]:
        """Get fallback product information when API is unavailable."""
        
        # Basic product information mapping
        product_info = {
            MODISProduct.MOD04_L2: {
                "title": "MODIS/Terra Aerosol 5-Min L2 Swath 10km",
                "description": "Level-2 aerosol product containing aerosol optical thickness and other aerosol properties",
                "platform": "Terra",
                "instrument": "MODIS",
                "temporal_resolution": "5 minutes",
                "spatial_resolution": "10 km"
            },
            MODISProduct.MOD06_L2: {
                "title": "MODIS/Terra Clouds 5-Min L2 Swath 1km and 5km", 
                "description": "Level-2 cloud product containing cloud optical and physical properties",
                "platform": "Terra",
                "instrument": "MODIS",
                "temporal_resolution": "5 minutes",
                "spatial_resolution": "1 km, 5 km"
            },
            MODISProduct.MOD11A1: {
                "title": "MODIS/Terra Land Surface Temperature/Emissivity Daily L3 Global 1km SIN Grid",
                "description": "Daily land surface temperature and emissivity values in a 1200 x 1200 kilometer grid",
                "platform": "Terra", 
                "instrument": "MODIS",
                "temporal_resolution": "Daily",
                "spatial_resolution": "1 km"
            }
        }
        
        return product_info.get(product, {
            "title": f"{product.value} Product",
            "description": f"MODIS {product.value} data product",
            "platform": "Terra/Aqua",
            "instrument": "MODIS"
        })
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get MODAPS/LAADS service status and statistics."""
        
        return {
            "service": "NASA MODAPS/LAADS DAAC",
            "base_url": self.base_url,
            "api_url": self.api_url,
            "authenticated": self.api_key is not None,
            "active_requests": len(self.active_requests),
            "completed_requests": len(self.request_history),
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "rate_limit_remaining": self.rate_limiter.get_remaining_requests(),
            "session_active": self.session is not None
        }


# Global MODAPS/LAADS service instance
_modaps_service: Optional[MODAPSLAADSService] = None


async def get_modaps_service() -> MODAPSLAADSService:
    """Get or create the global MODAPS/LAADS service."""
    global _modaps_service
    
    if _modaps_service is None:
        _modaps_service = MODAPSLAADSService()
        await _modaps_service.initialize()
    
    return _modaps_service


async def close_modaps_service():
    """Close the global MODAPS/LAADS service."""
    global _modaps_service
    
    if _modaps_service:
        await _modaps_service.close()
        _modaps_service = None