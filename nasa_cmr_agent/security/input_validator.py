"""
Comprehensive Input Validation and Sanitization Framework.

Provides robust protection against injection attacks, XSS, and malicious input
for the NASA CMR Agent system with industry best practices.
"""

import re
import html
import json
import urllib.parse
from typing import Dict, Any, List, Optional, Union, Set
from dataclasses import dataclass
from enum import Enum
import structlog
from datetime import datetime, timezone

logger = structlog.get_logger(__name__)


class ValidationSeverity(Enum):
    """Validation result severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Input validation result."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    sanitized_value: Optional[Any] = None
    validation_rules_applied: List[str] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.validation_rules_applied is None:
            self.validation_rules_applied = []
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class SecurityInputValidator:
    """Comprehensive input validation and sanitization system."""
    
    def __init__(self):
        self.validation_stats = {
            "total_validations": 0,
            "blocked_inputs": 0,
            "sanitized_inputs": 0,
            "threats_detected": 0
        }
        
        # SQL Injection patterns
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
            r"(UNION\s+ALL\s+SELECT)",
            r"(--|\#|\/\*|\*\/)",
            r"(\'\s*(OR|AND)\s*\'\s*=\s*\')",
            r"(\d\s*(OR|AND)\s*\d\s*=\s*\d)",
            r"(EXEC\s*\(\s*@)",
            r"(sp_executesql)",
            r"(xp_cmdshell)"
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"onload\s*=",
            r"onerror\s*=",
            r"onclick\s*=",
            r"onmouseover\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>"
        ]
        
        # Command injection patterns
        self.command_injection_patterns = [
            r"(\||&|;|\$\(|\`)",
            r"(sh\s|bash\s|cmd\s|powershell\s)",
            r"(rm\s|del\s|format\s)",
            r"(wget\s|curl\s|nc\s|netcat\s)",
            r"(eval\s*\(|exec\s*\()"
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r"(\.\./|\.\.\\)",
            r"(/etc/passwd|/etc/shadow)",
            r"(windows/system32)",
            r"(\.\.%2F|\.\.%5C)"
        ]
        
        # Sensitive data patterns
        self.sensitive_patterns = {
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "api_key": r"\b[A-Za-z0-9]{32,}\b",
            "password_field": r"(password|passwd|pwd)\s*[:=]\s*\S+",
        }
        
        # Allowed characters for different input types
        self.allowed_chars = {
            "alphanumeric": re.compile(r"^[a-zA-Z0-9\s]*$"),
            "query_safe": re.compile(r"^[a-zA-Z0-9\s\-_.()]*$"),
            "filename": re.compile(r"^[a-zA-Z0-9\-_.]*$"),
            "coordinates": re.compile(r"^[-+]?[0-9]*\.?[0-9]+$"),
            "datetime": re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:\d{2}|Z)?$")
        }
    
    def validate_query_input(self, query: str, max_length: int = 1000) -> ValidationResult:
        """Validate NASA CMR search query input."""
        self.validation_stats["total_validations"] += 1
        rules_applied = []
        
        # Basic validation
        if not isinstance(query, str):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Query must be a string",
                validation_rules_applied=["type_check"]
            )
        
        # Length validation
        if len(query) > max_length:
            self.validation_stats["blocked_inputs"] += 1
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Query exceeds maximum length of {max_length} characters",
                validation_rules_applied=["length_check"]
            )
        rules_applied.append("length_check")
        
        # SQL Injection detection
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                self.validation_stats["threats_detected"] += 1
                self.validation_stats["blocked_inputs"] += 1
                logger.warning(f"SQL injection pattern detected in query: {pattern}")
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    message="Potential SQL injection detected",
                    validation_rules_applied=rules_applied + ["sql_injection_check"]
                )
        rules_applied.append("sql_injection_check")
        
        # XSS detection
        for pattern in self.xss_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                self.validation_stats["threats_detected"] += 1
                self.validation_stats["blocked_inputs"] += 1
                logger.warning(f"XSS pattern detected in query: {pattern}")
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    message="Potential XSS attack detected",
                    validation_rules_applied=rules_applied + ["xss_check"]
                )
        rules_applied.append("xss_check")
        
        # Command injection detection
        for pattern in self.command_injection_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                self.validation_stats["threats_detected"] += 1
                self.validation_stats["blocked_inputs"] += 1
                logger.warning(f"Command injection pattern detected in query: {pattern}")
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    message="Potential command injection detected",
                    validation_rules_applied=rules_applied + ["command_injection_check"]
                )
        rules_applied.append("command_injection_check")
        
        # Sanitize the query
        sanitized_query = self._sanitize_query(query)
        if sanitized_query != query:
            self.validation_stats["sanitized_inputs"] += 1
            rules_applied.append("sanitization")
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Query validation passed",
            sanitized_value=sanitized_query,
            validation_rules_applied=rules_applied
        )
    
    def validate_coordinates(self, lat: Union[str, float], lon: Union[str, float]) -> ValidationResult:
        """Validate geographic coordinates."""
        self.validation_stats["total_validations"] += 1
        rules_applied = ["coordinate_validation"]
        
        try:
            # Convert to float if string
            lat_float = float(lat) if isinstance(lat, str) else lat
            lon_float = float(lon) if isinstance(lon, str) else lon
            
            # Validate latitude range
            if not -90 <= lat_float <= 90:
                self.validation_stats["blocked_inputs"] += 1
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Latitude {lat_float} out of valid range (-90 to 90)",
                    validation_rules_applied=rules_applied
                )
            
            # Validate longitude range
            if not -180 <= lon_float <= 180:
                self.validation_stats["blocked_inputs"] += 1
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Longitude {lon_float} out of valid range (-180 to 180)",
                    validation_rules_applied=rules_applied
                )
            
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="Coordinates validation passed",
                sanitized_value={"lat": lat_float, "lon": lon_float},
                validation_rules_applied=rules_applied
            )
            
        except (ValueError, TypeError) as e:
            self.validation_stats["blocked_inputs"] += 1
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid coordinate format: {e}",
                validation_rules_applied=rules_applied
            )
    
    def validate_json_input(self, json_data: Union[str, Dict], max_depth: int = 10, max_keys: int = 100) -> ValidationResult:
        """Validate JSON input with structure limits."""
        self.validation_stats["total_validations"] += 1
        rules_applied = ["json_validation"]
        
        # Parse JSON if string
        if isinstance(json_data, str):
            try:
                parsed_data = json.loads(json_data)
            except json.JSONDecodeError as e:
                self.validation_stats["blocked_inputs"] += 1
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid JSON format: {e}",
                    validation_rules_applied=rules_applied
                )
        else:
            parsed_data = json_data
        
        # Check structure limits
        depth_check = self._check_json_depth(parsed_data, max_depth)
        if not depth_check["valid"]:
            self.validation_stats["blocked_inputs"] += 1
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"JSON depth exceeds limit of {max_depth}",
                validation_rules_applied=rules_applied + ["depth_check"]
            )
        
        key_count = self._count_json_keys(parsed_data)
        if key_count > max_keys:
            self.validation_stats["blocked_inputs"] += 1
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"JSON key count {key_count} exceeds limit of {max_keys}",
                validation_rules_applied=rules_applied + ["key_count_check"]
            )
        
        # Sanitize JSON values
        sanitized_data = self._sanitize_json_values(parsed_data)
        if sanitized_data != parsed_data:
            self.validation_stats["sanitized_inputs"] += 1
            rules_applied.append("json_sanitization")
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="JSON validation passed",
            sanitized_value=sanitized_data,
            validation_rules_applied=rules_applied
        )
    
    def detect_sensitive_data(self, text: str) -> ValidationResult:
        """Detect potentially sensitive data in input."""
        self.validation_stats["total_validations"] += 1
        detected_types = []
        rules_applied = ["sensitive_data_detection"]
        
        for data_type, pattern in self.sensitive_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                detected_types.append(data_type)
        
        if detected_types:
            self.validation_stats["threats_detected"] += 1
            logger.warning(f"Sensitive data detected: {detected_types}")
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Potential sensitive data detected: {', '.join(detected_types)}",
                validation_rules_applied=rules_applied
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="No sensitive data detected",
            validation_rules_applied=rules_applied
        )
    
    def validate_file_path(self, file_path: str, allowed_extensions: Set[str] = None) -> ValidationResult:
        """Validate file path for path traversal and allowed extensions."""
        self.validation_stats["total_validations"] += 1
        rules_applied = ["file_path_validation"]
        
        if allowed_extensions is None:
            allowed_extensions = {".json", ".txt", ".csv", ".log"}
        
        # Path traversal detection
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, file_path, re.IGNORECASE):
                self.validation_stats["threats_detected"] += 1
                self.validation_stats["blocked_inputs"] += 1
                logger.warning(f"Path traversal pattern detected: {pattern}")
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    message="Path traversal attempt detected",
                    validation_rules_applied=rules_applied + ["path_traversal_check"]
                )
        
        # Extension validation
        if allowed_extensions:
            file_ext = "." + file_path.split(".")[-1].lower() if "." in file_path else ""
            if file_ext not in allowed_extensions:
                self.validation_stats["blocked_inputs"] += 1
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"File extension '{file_ext}' not in allowed list: {allowed_extensions}",
                    validation_rules_applied=rules_applied + ["extension_check"]
                )
        
        # Sanitize file path
        sanitized_path = self._sanitize_file_path(file_path)
        if sanitized_path != file_path:
            self.validation_stats["sanitized_inputs"] += 1
            rules_applied.append("path_sanitization")
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="File path validation passed",
            sanitized_value=sanitized_path,
            validation_rules_applied=rules_applied
        )
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize query input."""
        # HTML encode special characters
        sanitized = html.escape(query)
        
        # URL decode any encoded characters
        sanitized = urllib.parse.unquote(sanitized)
        
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', sanitized)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized.strip())
        
        return sanitized
    
    def _sanitize_json_values(self, data: Any) -> Any:
        """Recursively sanitize JSON values."""
        if isinstance(data, dict):
            return {key: self._sanitize_json_values(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_json_values(item) for item in data]
        elif isinstance(data, str):
            return self._sanitize_query(data)
        else:
            return data
    
    def _sanitize_file_path(self, file_path: str) -> str:
        """Sanitize file path."""
        # Remove path traversal sequences
        sanitized = re.sub(r'\.\.[\\/]', '', file_path)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Normalize path separators
        sanitized = sanitized.replace('\\', '/')
        
        return sanitized
    
    def _check_json_depth(self, data: Any, max_depth: int, current_depth: int = 0) -> Dict[str, Any]:
        """Check JSON nesting depth."""
        if current_depth >= max_depth:
            return {"valid": False, "depth": current_depth}
        
        if isinstance(data, dict):
            for value in data.values():
                result = self._check_json_depth(value, max_depth, current_depth + 1)
                if not result["valid"]:
                    return result
        elif isinstance(data, list):
            for item in data:
                result = self._check_json_depth(item, max_depth, current_depth + 1)
                if not result["valid"]:
                    return result
        
        return {"valid": True, "depth": current_depth}
    
    def _count_json_keys(self, data: Any, count: int = 0) -> int:
        """Count total keys in JSON structure."""
        if isinstance(data, dict):
            count += len(data)
            for value in data.values():
                count = self._count_json_keys(value, count)
        elif isinstance(data, list):
            for item in data:
                count = self._count_json_keys(item, count)
        
        return count
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.validation_stats["total_validations"]
        return {
            **self.validation_stats,
            "threat_detection_rate": self.validation_stats["threats_detected"] / total if total > 0 else 0,
            "block_rate": self.validation_stats["blocked_inputs"] / total if total > 0 else 0,
            "sanitization_rate": self.validation_stats["sanitized_inputs"] / total if total > 0 else 0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Global validator instance
_validator: Optional[SecurityInputValidator] = None


def get_input_validator() -> SecurityInputValidator:
    """Get or create the global input validator."""
    global _validator
    
    if _validator is None:
        _validator = SecurityInputValidator()
    
    return _validator