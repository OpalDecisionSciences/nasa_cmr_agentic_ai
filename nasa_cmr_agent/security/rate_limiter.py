"""
Advanced Rate Limiting and DDoS Protection System.

Provides multi-tier rate limiting, DDoS protection, and adaptive throttling
for the NASA CMR Agent system with intelligent traffic analysis.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import structlog

logger = structlog.get_logger(__name__)


class LimitType(Enum):
    """Types of rate limits."""
    REQUESTS_PER_SECOND = "requests_per_second"
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    REQUESTS_PER_DAY = "requests_per_day"
    CONCURRENT_CONNECTIONS = "concurrent_connections"
    BANDWIDTH_PER_SECOND = "bandwidth_per_second"
    QUERIES_PER_MINUTE = "queries_per_minute"


class ThreatLevel(Enum):
    """DDoS threat levels."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RateLimit:
    """Rate limit configuration."""
    limit_type: LimitType
    limit: int
    window_seconds: int
    burst_allowance: int = 0
    penalty_seconds: int = 60
    whitelist_multiplier: float = 1.0


@dataclass
class ClientInfo:
    """Client information for rate limiting."""
    client_id: str
    ip_address: str
    user_agent: str
    first_seen: str
    last_seen: str
    total_requests: int = 0
    blocked_requests: int = 0
    reputation_score: float = 100.0  # 0-100, lower is worse
    is_whitelisted: bool = False
    is_blacklisted: bool = False
    threat_level: ThreatLevel = ThreatLevel.NORMAL


@dataclass
class RateLimitResult:
    """Rate limit check result."""
    allowed: bool
    limit_type: LimitType
    current_count: int
    limit: int
    reset_time: float
    retry_after: Optional[int] = None
    penalty_applied: bool = False
    threat_level: ThreatLevel = ThreatLevel.NORMAL


class TrafficAnalyzer:
    """Analyzes traffic patterns for DDoS detection."""
    
    def __init__(self, analysis_window: int = 300):  # 5 minutes
        self.analysis_window = analysis_window
        self.traffic_samples = deque(maxlen=1000)
        self.baseline_established = False
        self.baseline_rps = 0.0
        self.anomaly_threshold = 3.0  # Standard deviations
    
    def record_request(self, timestamp: float, client_id: str, size_bytes: int = 0):
        """Record a request for traffic analysis."""
        self.traffic_samples.append({
            "timestamp": timestamp,
            "client_id": client_id,
            "size_bytes": size_bytes
        })
    
    def analyze_traffic_patterns(self) -> Dict[str, Any]:
        """Analyze current traffic patterns."""
        now = time.time()
        recent_samples = [
            s for s in self.traffic_samples 
            if now - s["timestamp"] <= self.analysis_window
        ]
        
        if not recent_samples:
            return {"threat_level": ThreatLevel.NORMAL, "analysis": "No traffic"}
        
        # Calculate requests per second
        current_rps = len(recent_samples) / self.analysis_window
        
        # Analyze client distribution
        client_counts = defaultdict(int)
        total_bytes = 0
        
        for sample in recent_samples:
            client_counts[sample["client_id"]] += 1
            total_bytes += sample["size_bytes"]
        
        # Calculate statistics
        unique_clients = len(client_counts)
        max_requests_per_client = max(client_counts.values()) if client_counts else 0
        avg_requests_per_client = current_rps / unique_clients if unique_clients > 0 else 0
        
        # DDoS indicators
        threat_indicators = []
        threat_level = ThreatLevel.NORMAL
        
        # High request rate
        if not self.baseline_established:
            self.baseline_rps = current_rps
            self.baseline_established = True
        elif current_rps > self.baseline_rps * self.anomaly_threshold:
            threat_indicators.append("high_request_rate")
            threat_level = ThreatLevel.HIGH
        
        # Client concentration (few clients making many requests)
        if unique_clients > 0:
            concentration_ratio = max_requests_per_client / len(recent_samples)
            if concentration_ratio > 0.7:  # Single client > 70% of traffic
                threat_indicators.append("client_concentration")
                if threat_level == ThreatLevel.NORMAL:
                    threat_level = ThreatLevel.ELEVATED
        
        # Bandwidth analysis
        bandwidth_mbps = (total_bytes / self.analysis_window) / (1024 * 1024)
        if bandwidth_mbps > 100:  # > 100 MB/s
            threat_indicators.append("high_bandwidth")
            threat_level = ThreatLevel.HIGH
        
        # Request pattern uniformity (potential bot traffic)
        if unique_clients > 10:
            request_variance = self._calculate_request_variance(client_counts.values())
            if request_variance < 0.1:  # Very uniform distribution
                threat_indicators.append("uniform_pattern")
                if threat_level == ThreatLevel.NORMAL:
                    threat_level = ThreatLevel.ELEVATED
        
        return {
            "threat_level": threat_level,
            "current_rps": current_rps,
            "baseline_rps": self.baseline_rps,
            "unique_clients": unique_clients,
            "max_requests_per_client": max_requests_per_client,
            "avg_requests_per_client": avg_requests_per_client,
            "bandwidth_mbps": bandwidth_mbps,
            "threat_indicators": threat_indicators,
            "analysis_timestamp": now
        }
    
    def _calculate_request_variance(self, values: List[int]) -> float:
        """Calculate variance in request counts."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance / (mean ** 2) if mean > 0 else 0.0


class AdvancedRateLimiter:
    """Advanced rate limiting and DDoS protection system."""
    
    def __init__(self):
        self.limits: Dict[str, List[RateLimit]] = {}
        self.client_data: Dict[str, ClientInfo] = {}
        self.request_history: Dict[str, Dict[LimitType, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=10000))
        )
        self.blocked_ips: Dict[str, float] = {}  # IP -> unblock_time
        self.concurrent_connections: Dict[str, int] = defaultdict(int)
        
        # Traffic analysis
        self.traffic_analyzer = TrafficAnalyzer()
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "ddos_attacks_detected": 0,
            "clients_blocked": 0,
            "adaptive_limits_applied": 0
        }
        
        # Initialize default limits
        self._initialize_default_limits()
        
        # Cleanup task
        self._cleanup_task = None
    
    def _initialize_default_limits(self):
        """Initialize default rate limits."""
        self.set_limits("default", [
            RateLimit(LimitType.REQUESTS_PER_SECOND, 10, 1, burst_allowance=5),
            RateLimit(LimitType.REQUESTS_PER_MINUTE, 300, 60, burst_allowance=50),
            RateLimit(LimitType.REQUESTS_PER_HOUR, 5000, 3600, burst_allowance=500),
            RateLimit(LimitType.CONCURRENT_CONNECTIONS, 50, 0),
            RateLimit(LimitType.QUERIES_PER_MINUTE, 100, 60, burst_allowance=20)
        ])
        
        # NASA CMR API specific limits
        self.set_limits("cmr_api", [
            RateLimit(LimitType.REQUESTS_PER_SECOND, 5, 1, burst_allowance=2),
            RateLimit(LimitType.REQUESTS_PER_MINUTE, 100, 60, burst_allowance=20),
            RateLimit(LimitType.QUERIES_PER_MINUTE, 50, 60, burst_allowance=10)
        ])
        
        # Database operation limits
        self.set_limits("database", [
            RateLimit(LimitType.REQUESTS_PER_SECOND, 20, 1, burst_allowance=10),
            RateLimit(LimitType.REQUESTS_PER_MINUTE, 600, 60, burst_allowance=100)
        ])
    
    def set_limits(self, service: str, limits: List[RateLimit]):
        """Set rate limits for a service."""
        self.limits[service] = limits
        logger.info(f"Rate limits configured for service: {service}")
    
    async def check_rate_limit(self, client_id: str, service: str = "default",
                             limit_type: LimitType = LimitType.REQUESTS_PER_SECOND,
                             ip_address: str = None, user_agent: str = None,
                             request_size_bytes: int = 0) -> RateLimitResult:
        """Check if request is allowed under rate limits."""
        self.stats["total_requests"] += 1
        now = time.time()
        
        # Record traffic for analysis
        if ip_address:
            self.traffic_analyzer.record_request(now, client_id, request_size_bytes)
        
        # Check if IP is blocked
        if ip_address and ip_address in self.blocked_ips:
            if now < self.blocked_ips[ip_address]:
                self.stats["blocked_requests"] += 1
                return RateLimitResult(
                    allowed=False,
                    limit_type=limit_type,
                    current_count=0,
                    limit=0,
                    reset_time=self.blocked_ips[ip_address],
                    retry_after=int(self.blocked_ips[ip_address] - now),
                    penalty_applied=True,
                    threat_level=ThreatLevel.HIGH
                )
            else:
                # Unblock IP
                del self.blocked_ips[ip_address]
        
        # Update client info
        await self._update_client_info(client_id, ip_address, user_agent)
        
        # Get applicable limits
        limits = self.limits.get(service, self.limits["default"])
        applicable_limit = None
        
        for limit in limits:
            if limit.limit_type == limit_type:
                applicable_limit = limit
                break
        
        if not applicable_limit:
            # No limit configured, allow request
            return RateLimitResult(
                allowed=True,
                limit_type=limit_type,
                current_count=0,
                limit=float('inf'),
                reset_time=now
            )
        
        # Check traffic patterns for DDoS
        traffic_analysis = self.traffic_analyzer.analyze_traffic_patterns()
        threat_level = traffic_analysis["threat_level"]
        
        # Apply adaptive limits based on threat level
        effective_limit = self._calculate_effective_limit(applicable_limit, threat_level, client_id)
        
        # Get request history for this client and limit type
        history = self.request_history[client_id][limit_type]
        
        # Clean old entries
        cutoff_time = now - applicable_limit.window_seconds
        while history and history[0] <= cutoff_time:
            history.popleft()
        
        # Check if limit exceeded
        current_count = len(history)
        
        # Check for burst allowance
        if current_count >= effective_limit:
            # Check if we can use burst allowance
            recent_cutoff = now - min(applicable_limit.window_seconds / 10, 5)  # Last 10% of window or 5 sec
            recent_requests = sum(1 for timestamp in history if timestamp >= recent_cutoff)
            
            if recent_requests >= applicable_limit.burst_allowance:
                # Rate limit exceeded
                await self._handle_rate_limit_exceeded(
                    client_id, ip_address, applicable_limit, threat_level
                )
                
                self.stats["blocked_requests"] += 1
                return RateLimitResult(
                    allowed=False,
                    limit_type=limit_type,
                    current_count=current_count,
                    limit=effective_limit,
                    reset_time=now + applicable_limit.window_seconds,
                    retry_after=applicable_limit.penalty_seconds if applicable_limit.penalty_seconds > 0 else None,
                    threat_level=threat_level
                )
        
        # Record request
        history.append(now)
        
        # Update concurrent connections for connection-based limits
        if limit_type == LimitType.CONCURRENT_CONNECTIONS:
            self.concurrent_connections[client_id] += 1
        
        return RateLimitResult(
            allowed=True,
            limit_type=limit_type,
            current_count=current_count + 1,
            limit=effective_limit,
            reset_time=now + applicable_limit.window_seconds,
            threat_level=threat_level
        )
    
    async def release_connection(self, client_id: str):
        """Release a concurrent connection."""
        if client_id in self.concurrent_connections:
            self.concurrent_connections[client_id] = max(0, self.concurrent_connections[client_id] - 1)
    
    def _calculate_effective_limit(self, limit: RateLimit, threat_level: ThreatLevel, 
                                 client_id: str) -> int:
        """Calculate effective limit based on threat level and client reputation."""
        base_limit = limit.limit
        
        # Get client info
        client = self.client_data.get(client_id)
        if client and client.is_whitelisted:
            base_limit = int(base_limit * max(limit.whitelist_multiplier, 2.0))  # At least 2x for whitelisted
        
        # Apply threat-based adjustments
        if threat_level == ThreatLevel.ELEVATED:
            base_limit = int(base_limit * 0.8)  # 20% reduction
        elif threat_level == ThreatLevel.HIGH:
            base_limit = int(base_limit * 0.5)  # 50% reduction
        elif threat_level == ThreatLevel.CRITICAL:
            base_limit = int(base_limit * 0.2)  # 80% reduction
        
        # Apply reputation-based adjustments
        if client:
            reputation_factor = client.reputation_score / 100.0
            base_limit = int(base_limit * reputation_factor)
        
        return max(1, base_limit)  # Always allow at least 1 request
    
    async def _update_client_info(self, client_id: str, ip_address: str = None, 
                                user_agent: str = None):
        """Update client information."""
        now = datetime.now(timezone.utc).isoformat()
        
        if client_id not in self.client_data:
            self.client_data[client_id] = ClientInfo(
                client_id=client_id,
                ip_address=ip_address or "unknown",
                user_agent=user_agent or "unknown",
                first_seen=now,
                last_seen=now
            )
        else:
            client = self.client_data[client_id]
            client.last_seen = now
            client.total_requests += 1
            
            # Update IP and user agent if provided
            if ip_address:
                client.ip_address = ip_address
            if user_agent:
                client.user_agent = user_agent
    
    async def _handle_rate_limit_exceeded(self, client_id: str, ip_address: str,
                                        limit: RateLimit, threat_level: ThreatLevel):
        """Handle rate limit exceeded."""
        client = self.client_data.get(client_id)
        if client:
            client.blocked_requests += 1
            
            # Reduce reputation score
            reputation_penalty = 5.0
            if threat_level == ThreatLevel.HIGH:
                reputation_penalty = 15.0
            elif threat_level == ThreatLevel.CRITICAL:
                reputation_penalty = 30.0
            
            client.reputation_score = max(0, client.reputation_score - reputation_penalty)
            
            # Block IP if reputation is very low or threat level is critical
            if (client.reputation_score < 20 or threat_level == ThreatLevel.CRITICAL):
                if ip_address:
                    block_duration = limit.penalty_seconds
                    if threat_level == ThreatLevel.CRITICAL:
                        block_duration *= 5  # 5x longer for critical threats
                    
                    self.blocked_ips[ip_address] = time.time() + block_duration
                    self.stats["clients_blocked"] += 1
                    
                    logger.warning(f"Blocked IP {ip_address} for {block_duration} seconds", 
                                 client_id=client_id, threat_level=threat_level.value)
        
        # Log security event
        try:
            from .audit_logger import get_audit_logger
            audit_logger = get_audit_logger()
            
            await audit_logger.log_rate_limiting_event(
                exceeded=True,
                limit_type=limit.limit_type.value,
                current_count=client.blocked_requests if client else 0,
                limit=limit.limit,
                ip_address=ip_address or "unknown",
                user_id=client_id
            )
        except Exception as e:
            logger.debug(f"Failed to log rate limit event: {e}")
    
    def whitelist_client(self, client_id: str, ip_address: str = None):
        """Add client to whitelist."""
        if client_id not in self.client_data:
            self.client_data[client_id] = ClientInfo(
                client_id=client_id,
                ip_address=ip_address or "unknown",
                user_agent="whitelisted",
                first_seen=datetime.now(timezone.utc).isoformat(),
                last_seen=datetime.now(timezone.utc).isoformat()
            )
        
        self.client_data[client_id].is_whitelisted = True
        self.client_data[client_id].reputation_score = 100.0
        
        logger.info(f"Whitelisted client: {client_id}")
    
    def blacklist_client(self, client_id: str, ip_address: str = None, duration_seconds: int = 3600):
        """Add client to blacklist."""
        if client_id not in self.client_data:
            self.client_data[client_id] = ClientInfo(
                client_id=client_id,
                ip_address=ip_address or "unknown",
                user_agent="blacklisted",
                first_seen=datetime.now(timezone.utc).isoformat(),
                last_seen=datetime.now(timezone.utc).isoformat()
            )
        
        self.client_data[client_id].is_blacklisted = True
        self.client_data[client_id].reputation_score = 0.0
        
        # Block IP if provided
        if ip_address:
            self.blocked_ips[ip_address] = time.time() + duration_seconds
        
        logger.warning(f"Blacklisted client: {client_id} for {duration_seconds} seconds")
    
    def get_client_stats(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific client."""
        client = self.client_data.get(client_id)
        if not client:
            return None
        
        return {
            "client_id": client.client_id,
            "ip_address": client.ip_address,
            "first_seen": client.first_seen,
            "last_seen": client.last_seen,
            "total_requests": client.total_requests,
            "blocked_requests": client.blocked_requests,
            "reputation_score": client.reputation_score,
            "is_whitelisted": client.is_whitelisted,
            "is_blacklisted": client.is_blacklisted,
            "threat_level": client.threat_level.value
        }
    
    def get_traffic_analysis(self) -> Dict[str, Any]:
        """Get current traffic analysis."""
        return self.traffic_analyzer.analyze_traffic_patterns()
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        active_blocks = sum(1 for unblock_time in self.blocked_ips.values() 
                           if unblock_time > time.time())
        
        return {
            **self.stats,
            "active_clients": len(self.client_data),
            "whitelisted_clients": sum(1 for c in self.client_data.values() if c.is_whitelisted),
            "blacklisted_clients": sum(1 for c in self.client_data.values() if c.is_blacklisted),
            "active_ip_blocks": active_blocks,
            "concurrent_connections": sum(self.concurrent_connections.values()),
            "services_configured": len(self.limits),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def cleanup_expired_data(self):
        """Clean up expired rate limit data."""
        now = time.time()
        cleanup_cutoff = now - 3600  # 1 hour
        
        # Clean up request histories
        for client_history in self.request_history.values():
            for limit_type, history in client_history.items():
                while history and history[0] <= cleanup_cutoff:
                    history.popleft()
        
        # Clean up expired IP blocks
        expired_ips = [ip for ip, unblock_time in self.blocked_ips.items() 
                      if unblock_time <= now]
        for ip in expired_ips:
            del self.blocked_ips[ip]
        
        # Clean up old client data (keep for 7 days)
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        expired_clients = []
        
        for client_id, client in self.client_data.items():
            last_seen = datetime.fromisoformat(client.last_seen)
            if last_seen < week_ago and not client.is_whitelisted:
                expired_clients.append(client_id)
        
        for client_id in expired_clients:
            del self.client_data[client_id]
            if client_id in self.request_history:
                del self.request_history[client_id]
        
        logger.debug(f"Cleaned up {len(expired_ips)} expired IP blocks and "
                    f"{len(expired_clients)} old client records")
    
    async def start_background_tasks(self):
        """Start background maintenance tasks."""
        async def cleanup_loop():
            while True:
                try:
                    await self.cleanup_expired_data()
                    await asyncio.sleep(300)  # Every 5 minutes
                except Exception as e:
                    logger.error(f"Rate limiter cleanup error: {e}")
                    await asyncio.sleep(60)
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def stop_background_tasks(self):
        """Stop background maintenance tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


# Global rate limiter instance
_rate_limiter: Optional[AdvancedRateLimiter] = None


def get_rate_limiter() -> AdvancedRateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = AdvancedRateLimiter()
    
    return _rate_limiter