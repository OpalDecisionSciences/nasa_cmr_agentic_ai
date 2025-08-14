"""
Port Monitoring and Conflict Detection Service.

Provides port availability checking, conflict detection, and graceful fallback
for monitoring services to prevent deployment issues.
"""

import asyncio
import socket
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class PortStatus(Enum):
    """Port availability status."""
    AVAILABLE = "available"
    OCCUPIED = "occupied"
    RESTRICTED = "restricted"
    ERROR = "error"


@dataclass
class PortCheck:
    """Port availability check result."""
    port: int
    status: PortStatus
    service_name: Optional[str] = None
    process_info: Optional[str] = None
    check_timestamp: Optional[str] = None
    error_message: Optional[str] = None


class PortMonitorService:
    """Port monitoring and conflict detection service."""
    
    def __init__(self):
        self.default_ports = {
            "prometheus": 8000,
            "grafana": 3000,
            "redis": 6379,
            "neo4j_http": 7474,
            "neo4j_bolt": 7687,
            "weaviate": 8080,
            "weaviate_grpc": 50051
        }
        
        self.fallback_port_ranges = {
            "prometheus": (8001, 8010),
            "grafana": (3001, 3010),
            "monitoring": (9000, 9100)
        }
    
    async def check_port_availability(self, port: int, host: str = "localhost") -> PortCheck:
        """Check if a specific port is available."""
        try:
            # Create socket and try to bind
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            
            try:
                result = sock.connect_ex((host, port))
                if result == 0:
                    # Port is occupied
                    try:
                        # Try to get service info
                        service_info = await self._get_service_info(port)
                        return PortCheck(
                            port=port,
                            status=PortStatus.OCCUPIED,
                            service_name=service_info.get("name"),
                            process_info=service_info.get("process")
                        )
                    except:
                        return PortCheck(
                            port=port,
                            status=PortStatus.OCCUPIED
                        )
                else:
                    # Port appears available, try to bind to confirm
                    try:
                        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        test_sock.bind((host, port))
                        test_sock.close()
                        
                        return PortCheck(
                            port=port,
                            status=PortStatus.AVAILABLE
                        )
                    except PermissionError:
                        return PortCheck(
                            port=port,
                            status=PortStatus.RESTRICTED,
                            error_message="Permission denied - may require elevated privileges"
                        )
                    except Exception as bind_error:
                        return PortCheck(
                            port=port,
                            status=PortStatus.ERROR,
                            error_message=f"Bind test failed: {bind_error}"
                        )
            finally:
                sock.close()
                
        except Exception as e:
            return PortCheck(
                port=port,
                status=PortStatus.ERROR,
                error_message=str(e)
            )
    
    async def _get_service_info(self, port: int) -> Dict[str, str]:
        """Try to identify what service is running on a port."""
        service_info = {}
        
        try:
            # Check common service patterns
            if port == 8000:
                service_info["name"] = "prometheus"
            elif port == 3000:
                service_info["name"] = "grafana"
            elif port == 6379:
                service_info["name"] = "redis"
            elif port == 7474:
                service_info["name"] = "neo4j_http"
            elif port == 7687:
                service_info["name"] = "neo4j_bolt"
            elif port == 8080:
                service_info["name"] = "weaviate"
            elif port == 50051:
                service_info["name"] = "weaviate_grpc"
            else:
                service_info["name"] = f"unknown_service_port_{port}"
            
            # Try to get process info (platform-specific)
            try:
                import psutil
                for conn in psutil.net_connections():
                    if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                        try:
                            process = psutil.Process(conn.pid)
                            service_info["process"] = f"{process.name()} (PID: {conn.pid})"
                            break
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
            except ImportError:
                logger.debug("psutil not available for detailed process info")
                
        except Exception as e:
            logger.debug(f"Could not get service info for port {port}: {e}")
        
        return service_info
    
    async def check_multiple_ports(self, ports: List[int], host: str = "localhost") -> Dict[int, PortCheck]:
        """Check availability of multiple ports concurrently."""
        tasks = [self.check_port_availability(port, host) for port in ports]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        port_status = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                port_status[ports[i]] = PortCheck(
                    port=ports[i],
                    status=PortStatus.ERROR,
                    error_message=str(result)
                )
            else:
                port_status[ports[i]] = result
        
        return port_status
    
    async def find_available_port_in_range(self, start_port: int, end_port: int, 
                                         host: str = "localhost") -> Optional[int]:
        """Find the first available port in a given range."""
        for port in range(start_port, end_port + 1):
            check = await self.check_port_availability(port, host)
            if check.status == PortStatus.AVAILABLE:
                return port
        return None
    
    async def get_fallback_port(self, service_name: str, preferred_port: Optional[int] = None) -> Tuple[int, str]:
        """Get a fallback port for a service with conflict resolution."""
        # First try the preferred port
        if preferred_port:
            check = await self.check_port_availability(preferred_port)
            if check.status == PortStatus.AVAILABLE:
                return preferred_port, "preferred_port_available"
        
        # Try default port for the service
        default_port = self.default_ports.get(service_name)
        if default_port:
            check = await self.check_port_availability(default_port)
            if check.status == PortStatus.AVAILABLE:
                return default_port, "default_port_available"
        
        # Look for fallback port ranges
        if service_name in self.fallback_port_ranges:
            start, end = self.fallback_port_ranges[service_name]
            fallback_port = await self.find_available_port_in_range(start, end)
            if fallback_port:
                return fallback_port, f"fallback_port_range_{start}_{end}"
        
        # Generic fallback ranges
        generic_ranges = [
            (9000, 9100),  # General monitoring range
            (10000, 10100),  # High port range
            (8001, 8020)   # Alternative web service range
        ]
        
        for start, end in generic_ranges:
            fallback_port = await self.find_available_port_in_range(start, end)
            if fallback_port:
                return fallback_port, f"generic_fallback_{start}_{end}"
        
        raise RuntimeError(f"Could not find available port for service {service_name}")
    
    async def check_service_dependencies(self) -> Dict[str, Any]:
        """Check availability of all service dependency ports."""
        dependency_results = {}
        
        # Check all default service ports
        all_ports = list(self.default_ports.values())
        port_checks = await self.check_multiple_ports(all_ports)
        
        for service_name, port in self.default_ports.items():
            check = port_checks[port]
            dependency_results[service_name] = {
                "port": port,
                "status": check.status.value,
                "available": check.status == PortStatus.AVAILABLE,
                "service_info": check.service_name,
                "process_info": check.process_info,
                "error": check.error_message
            }
        
        # Summary statistics
        total_services = len(dependency_results)
        available_services = sum(1 for result in dependency_results.values() 
                               if result["available"])
        occupied_services = sum(1 for result in dependency_results.values() 
                              if result["status"] == "occupied")
        
        summary = {
            "total_services": total_services,
            "available_services": available_services,
            "occupied_services": occupied_services,
            "availability_rate": available_services / total_services if total_services > 0 else 0,
            "results": dependency_results
        }
        
        return summary
    
    async def generate_port_configuration(self, services: List[str]) -> Dict[str, Dict[str, Any]]:
        """Generate optimized port configuration avoiding conflicts."""
        configuration = {}
        
        for service in services:
            try:
                port, strategy = await self.get_fallback_port(service)
                configuration[service] = {
                    "port": port,
                    "strategy": strategy,
                    "status": "configured"
                }
                
                logger.info(f"Port configuration for {service}: {port} ({strategy})")
                
            except Exception as e:
                configuration[service] = {
                    "port": None,
                    "strategy": "failed",
                    "status": "error",
                    "error": str(e)
                }
                
                logger.error(f"Failed to configure port for {service}: {e}")
        
        return configuration
    
    def get_monitoring_recommendations(self, dependency_check: Dict[str, Any]) -> List[str]:
        """Generate monitoring setup recommendations based on port conflicts."""
        recommendations = []
        
        results = dependency_check.get("results", {})
        availability_rate = dependency_check.get("availability_rate", 0)
        
        if availability_rate < 0.5:
            recommendations.append(
                "⚠️ More than 50% of default ports are occupied. Consider using alternative ports."
            )
        
        # Service-specific recommendations
        for service, result in results.items():
            if result["status"] == "occupied":
                recommendations.append(
                    f"🔄 {service.title()} port {result['port']} is occupied "
                    f"by {result.get('service_info', 'unknown service')}. "
                    f"Consider using a different port or stopping the conflicting service."
                )
            elif result["status"] == "restricted":
                recommendations.append(
                    f"🔐 {service.title()} port {result['port']} requires elevated privileges. "
                    f"Run with appropriate permissions or use alternative port."
                )
            elif result["status"] == "error":
                recommendations.append(
                    f"❌ {service.title()} port {result['port']} check failed: {result.get('error', 'Unknown error')}"
                )
        
        if availability_rate > 0.8:
            recommendations.append(
                "✅ Most ports are available. Monitoring setup should proceed smoothly."
            )
        
        return recommendations


# Global port monitor instance
_port_monitor: Optional[PortMonitorService] = None


def get_port_monitor() -> PortMonitorService:
    """Get or create the global port monitor service."""
    global _port_monitor
    
    if _port_monitor is None:
        _port_monitor = PortMonitorService()
    
    return _port_monitor