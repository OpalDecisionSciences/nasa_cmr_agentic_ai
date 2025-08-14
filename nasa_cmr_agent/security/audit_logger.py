"""
Comprehensive Security Audit Logging System.

Provides enterprise-grade security event logging, threat detection,
and compliance reporting for the NASA CMR Agent system.
"""

import json
import hashlib
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone, timedelta
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)


class SecurityEventType(Enum):
    """Types of security events to audit."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    INPUT_VALIDATION = "input_validation"
    ENCRYPTION = "encryption"
    RATE_LIMITING = "rate_limiting"
    SYSTEM_ACCESS = "system_access"
    CONFIGURATION_CHANGE = "configuration_change"
    THREAT_DETECTION = "threat_detection"
    ERROR_SECURITY = "error_security"
    ADMIN_ACTION = "admin_action"


class SecurityEventSeverity(Enum):
    """Security event severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventStatus(Enum):
    """Security event status."""
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    BLOCKED = "blocked"


@dataclass
class SecurityEvent:
    """Security audit event."""
    event_id: str
    event_type: SecurityEventType
    severity: SecurityEventSeverity
    status: SecurityEventStatus
    timestamp: str
    source: str
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource: Optional[str]
    action: str
    description: str
    details: Dict[str, Any]
    risk_score: int  # 0-100
    compliance_tags: List[str]
    remediation_required: bool = False
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.event_id:
            self.event_id = self._generate_event_id()
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import secrets
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        random_suffix = secrets.token_hex(4)
        return f"SEC-{timestamp}-{random_suffix}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        data = self.to_dict()
        # Convert enums to string values for JSON serialization
        if hasattr(data['event_type'], 'value'):
            data['event_type'] = data['event_type'].value
        if hasattr(data['severity'], 'value'):
            data['severity'] = data['severity'].value
        if hasattr(data['status'], 'value'):
            data['status'] = data['status'].value
        return json.dumps(data, indent=2)


@dataclass
class ThreatPattern:
    """Security threat pattern definition."""
    pattern_id: str
    name: str
    description: str
    indicators: List[str]
    severity: SecurityEventSeverity
    auto_block: bool = False
    alert_threshold: int = 1


class SecurityAuditLogger:
    """Comprehensive security audit logging system."""
    
    def __init__(self, log_directory: str = "./logs/security", 
                 max_log_size_mb: int = 100, retention_days: int = 365):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        self.max_log_size_bytes = max_log_size_mb * 1024 * 1024
        self.retention_days = retention_days
        
        # Event storage and statistics
        self.recent_events: List[SecurityEvent] = []
        self.max_recent_events = 1000
        
        self.audit_stats = {
            "total_events": 0,
            "events_by_type": {},
            "events_by_severity": {},
            "threats_detected": 0,
            "blocked_actions": 0,
            "compliance_violations": 0
        }
        
        # Threat detection patterns
        self.threat_patterns = self._initialize_threat_patterns()
        
        # Real-time monitoring
        self._alert_callbacks = []
        
        # Compliance frameworks
        self.compliance_frameworks = {
            "SOX": ["data_access", "configuration_change", "admin_action"],
            "GDPR": ["data_access", "authentication", "data_export"],
            "HIPAA": ["data_access", "authentication", "encryption"],
            "PCI_DSS": ["authentication", "data_access", "encryption"],
            "NIST": ["all"]
        }
    
    def _initialize_threat_patterns(self) -> Dict[str, ThreatPattern]:
        """Initialize security threat detection patterns."""
        patterns = {
            "brute_force": ThreatPattern(
                pattern_id="BF-001",
                name="Brute Force Attack",
                description="Multiple failed authentication attempts",
                indicators=["multiple_auth_failures", "rapid_requests"],
                severity=SecurityEventSeverity.HIGH,
                auto_block=True,
                alert_threshold=5
            ),
            "sql_injection": ThreatPattern(
                pattern_id="SI-001", 
                name="SQL Injection Attempt",
                description="Malicious SQL patterns in input",
                indicators=["sql_keywords", "injection_patterns"],
                severity=SecurityEventSeverity.CRITICAL,
                auto_block=True,
                alert_threshold=1
            ),
            "xss_attempt": ThreatPattern(
                pattern_id="XSS-001",
                name="Cross-Site Scripting Attempt",
                description="Malicious script injection in input",
                indicators=["script_tags", "javascript_patterns"],
                severity=SecurityEventSeverity.HIGH,
                auto_block=True,
                alert_threshold=1
            ),
            "data_exfiltration": ThreatPattern(
                pattern_id="DE-001",
                name="Data Exfiltration Attempt", 
                description="Unusual data access patterns",
                indicators=["bulk_data_access", "off_hours_access"],
                severity=SecurityEventSeverity.CRITICAL,
                auto_block=False,
                alert_threshold=3
            ),
            "privilege_escalation": ThreatPattern(
                pattern_id="PE-001",
                name="Privilege Escalation Attempt",
                description="Unauthorized access to elevated functions",
                indicators=["admin_function_access", "permission_bypass"],
                severity=SecurityEventSeverity.CRITICAL,
                auto_block=True,
                alert_threshold=1
            )
        }
        
        return patterns
    
    async def log_security_event(self, event_type: SecurityEventType, 
                                status: SecurityEventStatus,
                                action: str, description: str,
                                severity: SecurityEventSeverity = SecurityEventSeverity.INFO,
                                user_id: Optional[str] = None,
                                session_id: Optional[str] = None,
                                ip_address: Optional[str] = None,
                                user_agent: Optional[str] = None,
                                resource: Optional[str] = None,
                                details: Optional[Dict[str, Any]] = None,
                                source: str = "nasa_cmr_agent") -> SecurityEvent:
        """Log a security event."""
        
        if details is None:
            details = {}
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(event_type, severity, status, details)
        
        # Determine compliance tags
        compliance_tags = self._get_compliance_tags(event_type)
        
        # Create security event
        event = SecurityEvent(
            event_id="",  # Will be auto-generated
            event_type=event_type,
            severity=severity,
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            description=description,
            details=details,
            risk_score=risk_score,
            compliance_tags=compliance_tags,
            remediation_required=severity in [SecurityEventSeverity.HIGH, SecurityEventSeverity.CRITICAL]
        )
        
        # Store event
        await self._store_event(event)
        
        # Update statistics
        self._update_statistics(event)
        
        # Threat detection analysis
        await self._analyze_for_threats(event)
        
        # Trigger alerts if needed
        await self._check_alert_conditions(event)
        
        logger.info(f"Security event logged: {event.event_id}", 
                   event_type=event_type.value, severity=severity.value)
        
        return event
    
    async def log_authentication_event(self, success: bool, user_id: str,
                                     ip_address: str, user_agent: str,
                                     details: Optional[Dict[str, Any]] = None) -> SecurityEvent:
        """Log authentication event."""
        status = SecurityEventStatus.SUCCESS if success else SecurityEventStatus.FAILURE
        severity = SecurityEventSeverity.INFO if success else SecurityEventSeverity.MEDIUM
        action = "login_success" if success else "login_failure"
        description = f"User authentication {'succeeded' if success else 'failed'}"
        
        return await self.log_security_event(
            event_type=SecurityEventType.AUTHENTICATION,
            status=status,
            action=action,
            description=description,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {}
        )
    
    async def log_data_access_event(self, user_id: str, resource: str,
                                  action: str, success: bool,
                                  details: Optional[Dict[str, Any]] = None) -> SecurityEvent:
        """Log data access event."""
        status = SecurityEventStatus.SUCCESS if success else SecurityEventStatus.FAILURE
        severity = SecurityEventSeverity.INFO if success else SecurityEventSeverity.MEDIUM
        description = f"Data access attempt: {action} on {resource}"
        
        return await self.log_security_event(
            event_type=SecurityEventType.DATA_ACCESS,
            status=status,
            action=action,
            description=description,
            severity=severity,
            user_id=user_id,
            resource=resource,
            details=details or {}
        )
    
    async def log_threat_detection(self, threat_type: str, indicators: List[str],
                                 details: Dict[str, Any],
                                 ip_address: Optional[str] = None,
                                 user_id: Optional[str] = None) -> SecurityEvent:
        """Log threat detection event."""
        self.audit_stats["threats_detected"] += 1
        
        return await self.log_security_event(
            event_type=SecurityEventType.THREAT_DETECTION,
            status=SecurityEventStatus.WARNING,
            action="threat_detected",
            description=f"Security threat detected: {threat_type}",
            severity=SecurityEventSeverity.HIGH,
            user_id=user_id,
            ip_address=ip_address,
            details={
                "threat_type": threat_type,
                "indicators": indicators,
                **details
            }
        )
    
    async def log_input_validation_event(self, validation_result: Dict[str, Any],
                                       user_id: Optional[str] = None,
                                       ip_address: Optional[str] = None) -> SecurityEvent:
        """Log input validation event."""
        is_valid = validation_result.get("is_valid", True)
        severity = validation_result.get("severity", "info")
        
        # Map severity string to enum
        severity_map = {
            "info": SecurityEventSeverity.INFO,
            "warning": SecurityEventSeverity.LOW,
            "error": SecurityEventSeverity.MEDIUM,
            "critical": SecurityEventSeverity.CRITICAL
        }
        
        status = SecurityEventStatus.SUCCESS if is_valid else SecurityEventStatus.BLOCKED
        severity_enum = severity_map.get(severity, SecurityEventSeverity.INFO)
        
        if not is_valid:
            self.audit_stats["blocked_actions"] += 1
        
        return await self.log_security_event(
            event_type=SecurityEventType.INPUT_VALIDATION,
            status=status,
            action="input_validation",
            description=f"Input validation {'passed' if is_valid else 'failed'}",
            severity=severity_enum,
            user_id=user_id,
            ip_address=ip_address,
            details=validation_result
        )
    
    async def log_rate_limiting_event(self, exceeded: bool, limit_type: str,
                                    current_count: int, limit: int,
                                    ip_address: str,
                                    user_id: Optional[str] = None) -> SecurityEvent:
        """Log rate limiting event."""
        status = SecurityEventStatus.BLOCKED if exceeded else SecurityEventStatus.SUCCESS
        severity = SecurityEventSeverity.MEDIUM if exceeded else SecurityEventSeverity.INFO
        action = "rate_limit_exceeded" if exceeded else "rate_limit_check"
        description = f"Rate limit {limit_type}: {current_count}/{limit}"
        
        if exceeded:
            self.audit_stats["blocked_actions"] += 1
        
        return await self.log_security_event(
            event_type=SecurityEventType.RATE_LIMITING,
            status=status,
            action=action,
            description=description,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            details={
                "limit_type": limit_type,
                "current_count": current_count,
                "limit": limit,
                "exceeded": exceeded
            }
        )
    
    async def _store_event(self, event: SecurityEvent):
        """Store security event to file."""
        # Add to recent events (in-memory)
        self.recent_events.append(event)
        if len(self.recent_events) > self.max_recent_events:
            self.recent_events.pop(0)
        
        # Write to log file
        log_file = self.log_directory / f"security-audit-{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        
        try:
            with open(log_file, 'a') as f:
                f.write(event.to_json() + '\n')
        except Exception as e:
            logger.error(f"Failed to write security event to log: {e}")
        
        # Rotate logs if needed
        await self._rotate_logs_if_needed(log_file)
    
    def _update_statistics(self, event: SecurityEvent):
        """Update audit statistics."""
        self.audit_stats["total_events"] += 1
        
        # Update by type
        event_type = event.event_type.value
        self.audit_stats["events_by_type"][event_type] = \
            self.audit_stats["events_by_type"].get(event_type, 0) + 1
        
        # Update by severity
        severity = event.severity.value
        self.audit_stats["events_by_severity"][severity] = \
            self.audit_stats["events_by_severity"].get(severity, 0) + 1
        
        # Update compliance violations
        if event.severity in [SecurityEventSeverity.HIGH, SecurityEventSeverity.CRITICAL]:
            self.audit_stats["compliance_violations"] += 1
    
    async def _analyze_for_threats(self, event: SecurityEvent):
        """Analyze event for threat patterns."""
        for pattern_id, pattern in self.threat_patterns.items():
            if await self._matches_threat_pattern(event, pattern):
                await self.log_threat_detection(
                    threat_type=pattern.name,
                    indicators=pattern.indicators,
                    details={
                        "pattern_id": pattern_id,
                        "original_event_id": event.event_id,
                        "auto_block": pattern.auto_block
                    },
                    ip_address=event.ip_address,
                    user_id=event.user_id
                )
    
    async def _matches_threat_pattern(self, event: SecurityEvent, pattern: ThreatPattern) -> bool:
        """Check if event matches a threat pattern."""
        # Simple pattern matching - in production, use more sophisticated ML-based detection
        
        if pattern.pattern_id == "BF-001":  # Brute force
            return (event.event_type == SecurityEventType.AUTHENTICATION and
                    event.status == SecurityEventStatus.FAILURE)
        
        elif pattern.pattern_id == "SI-001":  # SQL injection
            return (event.event_type == SecurityEventType.INPUT_VALIDATION and
                    "sql" in event.details.get("message", "").lower())
        
        elif pattern.pattern_id == "XSS-001":  # XSS attempt
            return (event.event_type == SecurityEventType.INPUT_VALIDATION and
                    "xss" in event.details.get("message", "").lower())
        
        elif pattern.pattern_id == "DE-001":  # Data exfiltration
            return (event.event_type == SecurityEventType.DATA_ACCESS and
                    event.details.get("bulk_access", False))
        
        elif pattern.pattern_id == "PE-001":  # Privilege escalation
            return (event.event_type == SecurityEventType.AUTHORIZATION and
                    event.status == SecurityEventStatus.FAILURE and
                    event.details.get("attempted_privilege", "") == "admin")
        
        return False
    
    async def _check_alert_conditions(self, event: SecurityEvent):
        """Check if event should trigger alerts."""
        # High/Critical severity events always trigger alerts
        if event.severity in [SecurityEventSeverity.HIGH, SecurityEventSeverity.CRITICAL]:
            await self._trigger_alert(event, "high_severity_event")
        
        # Multiple failed authentications from same IP
        if (event.event_type == SecurityEventType.AUTHENTICATION and 
            event.status == SecurityEventStatus.FAILURE):
            recent_failures = self._count_recent_events(
                event_type=SecurityEventType.AUTHENTICATION,
                status=SecurityEventStatus.FAILURE,
                ip_address=event.ip_address,
                minutes=15
            )
            if recent_failures >= 5:
                await self._trigger_alert(event, "brute_force_attempt")
        
        # Compliance violations
        if event.remediation_required:
            await self._trigger_alert(event, "compliance_violation")
    
    async def _trigger_alert(self, event: SecurityEvent, alert_type: str):
        """Trigger security alert."""
        alert_data = {
            "alert_type": alert_type,
            "event": event.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Call registered alert callbacks
        for callback in self._alert_callbacks:
            try:
                await callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        logger.warning(f"Security alert triggered: {alert_type}", 
                      event_id=event.event_id)
    
    def _calculate_risk_score(self, event_type: SecurityEventType, 
                            severity: SecurityEventSeverity,
                            status: SecurityEventStatus,
                            details: Dict[str, Any]) -> int:
        """Calculate risk score for event (0-100)."""
        base_scores = {
            SecurityEventSeverity.INFO: 10,
            SecurityEventSeverity.LOW: 25,
            SecurityEventSeverity.MEDIUM: 50,
            SecurityEventSeverity.HIGH: 75,
            SecurityEventSeverity.CRITICAL: 90
        }
        
        risk_score = base_scores.get(severity, 10)
        
        # Adjust based on event type
        type_modifiers = {
            SecurityEventType.THREAT_DETECTION: 20,
            SecurityEventType.AUTHENTICATION: 10,
            SecurityEventType.AUTHORIZATION: 15,
            SecurityEventType.INPUT_VALIDATION: 5
        }
        risk_score += type_modifiers.get(event_type, 0)
        
        # Adjust based on status
        if status == SecurityEventStatus.FAILURE:
            risk_score += 10
        elif status == SecurityEventStatus.BLOCKED:
            risk_score += 5
        
        return min(risk_score, 100)
    
    def _get_compliance_tags(self, event_type: SecurityEventType) -> List[str]:
        """Get compliance framework tags for event type."""
        tags = []
        event_type_str = event_type.value
        
        for framework, relevant_types in self.compliance_frameworks.items():
            if "all" in relevant_types or event_type_str in relevant_types:
                tags.append(framework)
        
        return tags
    
    def _count_recent_events(self, event_type: Optional[SecurityEventType] = None,
                           status: Optional[SecurityEventStatus] = None,
                           ip_address: Optional[str] = None,
                           user_id: Optional[str] = None,
                           minutes: int = 60) -> int:
        """Count recent events matching criteria."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        count = 0
        
        for event in self.recent_events:
            event_time = datetime.fromisoformat(event.timestamp)
            if event_time < cutoff_time:
                continue
            
            # Apply filters
            if event_type and event.event_type != event_type:
                continue
            if status and event.status != status:
                continue
            if ip_address and event.ip_address != ip_address:
                continue
            if user_id and event.user_id != user_id:
                continue
            
            count += 1
        
        return count
    
    async def _rotate_logs_if_needed(self, log_file: Path):
        """Rotate log file if it exceeds size limit."""
        try:
            if log_file.stat().st_size > self.max_log_size_bytes:
                # Rotate by adding timestamp
                rotated_name = f"{log_file.stem}-{datetime.now().strftime('%H%M%S')}{log_file.suffix}"
                rotated_file = log_file.parent / rotated_name
                log_file.rename(rotated_file)
                logger.info(f"Rotated security log: {rotated_file}")
        except Exception as e:
            logger.error(f"Failed to rotate log file: {e}")
    
    def add_alert_callback(self, callback):
        """Add callback function for security alerts."""
        self._alert_callbacks.append(callback)
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit statistics."""
        return {
            **self.audit_stats,
            "recent_events_count": len(self.recent_events),
            "threat_patterns_active": len(self.threat_patterns),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_recent_events(self, limit: int = 100, 
                         event_type: Optional[SecurityEventType] = None,
                         severity: Optional[SecurityEventSeverity] = None) -> List[SecurityEvent]:
        """Get recent security events with optional filtering."""
        filtered_events = []
        
        for event in reversed(self.recent_events):  # Most recent first
            if event_type and event.event_type != event_type:
                continue
            if severity and event.severity != severity:
                continue
            
            filtered_events.append(event)
            if len(filtered_events) >= limit:
                break
        
        return filtered_events
    
    async def generate_compliance_report(self, framework: str = "NIST",
                                       days: int = 30) -> Dict[str, Any]:
        """Generate compliance report for specified framework."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        relevant_events = []
        for event in self.recent_events:
            event_time = datetime.fromisoformat(event.timestamp)
            if (event_time >= cutoff_date and 
                framework in event.compliance_tags):
                relevant_events.append(event)
        
        # Generate report
        report = {
            "framework": framework,
            "report_period": f"{days} days",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_events": len(relevant_events),
            "events_by_severity": {},
            "events_by_type": {},
            "compliance_violations": [],
            "remediation_required": []
        }
        
        for event in relevant_events:
            # Count by severity
            severity = event.severity.value
            report["events_by_severity"][severity] = \
                report["events_by_severity"].get(severity, 0) + 1
            
            # Count by type
            event_type = event.event_type.value
            report["events_by_type"][event_type] = \
                report["events_by_type"].get(event_type, 0) + 1
            
            # Track violations
            if event.severity in [SecurityEventSeverity.HIGH, SecurityEventSeverity.CRITICAL]:
                report["compliance_violations"].append({
                    "event_id": event.event_id,
                    "timestamp": event.timestamp,
                    "severity": event.severity.value,
                    "description": event.description
                })
            
            if event.remediation_required:
                report["remediation_required"].append({
                    "event_id": event.event_id,
                    "timestamp": event.timestamp,
                    "action_required": event.description
                })
        
        return report


# Global audit logger instance
_audit_logger: Optional[SecurityAuditLogger] = None


def get_audit_logger(log_directory: str = "./logs/security") -> SecurityAuditLogger:
    """Get or create the global security audit logger."""
    global _audit_logger
    
    if _audit_logger is None:
        _audit_logger = SecurityAuditLogger(log_directory)
    
    return _audit_logger