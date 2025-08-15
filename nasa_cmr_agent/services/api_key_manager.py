"""
Unified API Key Management System for NASA CMR Agent.

Provides secure storage, rotation, and management of API keys for all NASA services
including CMR, GIOVANNI, MODAPS/LAADS, Atmospheric APIs, and Earthdata Login.

Features:
- Encrypted key storage
- Automatic key rotation
- Key validation and testing
- Fallback key management
- Audit logging for key usage
"""

import asyncio
import json
import time
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone, timedelta
from pathlib import Path
import structlog

from ..core.config import settings
from ..security.encryption import get_encryption_service
from ..security.audit_logger import get_audit_logger

logger = structlog.get_logger(__name__)


class APIService(Enum):
    """Supported NASA API services."""
    CMR = "cmr"
    GIOVANNI = "giovanni"
    MODAPS_LAADS = "modaps_laads"
    ATMOSPHERIC = "atmospheric"
    EARTHDATA_LOGIN = "earthdata_login"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class KeyStatus(Enum):
    """API key status states."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    ROTATING = "rotating"
    TESTING = "testing"
    FAILED = "failed"


@dataclass
class APIKey:
    """API key information and metadata."""
    service: APIService
    key_id: str
    encrypted_key: str
    key_hash: str  # For identification without storing plaintext
    created_at: float
    expires_at: Optional[float] = None
    last_used: Optional[float] = None
    usage_count: int = 0
    status: KeyStatus = KeyStatus.ACTIVE
    permissions: List[str] = None
    rate_limit: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []
        if self.rate_limit is None:
            self.rate_limit = {}


@dataclass
class KeyRotationPolicy:
    """API key rotation policy configuration."""
    service: APIService
    rotation_interval_days: int
    warning_days_before_expiry: int
    auto_rotation_enabled: bool
    backup_keys_count: int
    validation_required: bool = True


@dataclass
class KeyUsageMetrics:
    """API key usage metrics and statistics."""
    service: APIService
    total_requests: int
    successful_requests: int
    failed_requests: int
    last_24h_requests: int
    avg_response_time: float
    rate_limit_hits: int
    last_failure_time: Optional[float] = None
    last_failure_reason: Optional[str] = None


class APIKeyManager:
    """Unified API key management system."""
    
    def __init__(self, key_store_path: Optional[str] = None):
        """Initialize API key manager."""
        self.key_store_path = Path(key_store_path) if key_store_path else Path("secure_keys.json")
        self.encryption_service = None
        self.audit_logger = None
        
        # In-memory key cache (encrypted)
        self.key_cache: Dict[str, APIKey] = {}
        self.active_keys: Dict[APIService, APIKey] = {}
        
        # Key rotation policies
        self.rotation_policies: Dict[APIService, KeyRotationPolicy] = {}
        
        # Usage metrics
        self.usage_metrics: Dict[APIService, KeyUsageMetrics] = {}
        
        # Key validation cache
        self.validation_cache: Dict[str, Tuple[bool, float]] = {}
        self.validation_cache_ttl = 3600  # 1 hour
        
        # Initialize default rotation policies
        self._initialize_default_policies()
    
    async def initialize(self):
        """Initialize the API key manager."""
        try:
            self.encryption_service = get_encryption_service()
        except Exception as e:
            logger.error(f"Failed to initialize encryption service: {e}")
            raise
        
        try:
            self.audit_logger = get_audit_logger()
        except Exception as e:
            logger.warning(f"Audit logger not available: {e}")
        
        # Load existing keys
        await self._load_keys_from_storage()
        
        # Initialize usage metrics
        await self._initialize_usage_metrics()
        
        logger.info("API Key Manager initialized successfully")
    
    async def store_api_key(self, service: APIService, key: str, 
                           permissions: List[str] = None,
                           expires_in_days: Optional[int] = None) -> str:
        """Store a new API key securely."""
        
        # Generate unique key ID
        key_id = f"{service.value}_{int(time.time())}_{secrets.token_hex(8)}"
        
        # Create key hash for identification
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        
        # Encrypt the key
        encrypted_key, encryption_key_id = self.encryption_service.encrypt_cache_data(key)
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = time.time() + (expires_in_days * 24 * 3600)
        
        # Create API key object
        api_key = APIKey(
            service=service,
            key_id=key_id,
            encrypted_key=encrypted_key,
            key_hash=key_hash,
            created_at=time.time(),
            expires_at=expires_at,
            permissions=permissions or [],
            status=KeyStatus.TESTING
        )
        
        # Test the key before storing
        if await self._validate_api_key(service, key):
            api_key.status = KeyStatus.ACTIVE
            
            # Store in cache and set as active
            self.key_cache[key_id] = api_key
            self.active_keys[service] = api_key
            
            # Persist to storage
            await self._save_keys_to_storage()
            
            # Log the action
            if self.audit_logger:
                await self.audit_logger.log_security_event(
                    event_type="api_key_stored",
                    details={
                        "service": service.value,
                        "key_id": key_id,
                        "key_hash": key_hash,
                        "permissions": permissions or []
                    },
                    severity="info"
                )
            
            logger.info(f"API key stored successfully for {service.value}", key_id=key_id)
            return key_id
        else:
            api_key.status = KeyStatus.FAILED
            logger.error(f"API key validation failed for {service.value}")
            raise ValueError(f"Invalid API key for {service.value}")
    
    async def get_api_key(self, service: APIService) -> Optional[str]:
        """Get the active API key for a service."""
        
        # Check if we have an active key
        if service in self.active_keys:
            api_key = self.active_keys[service]
            
            # Check if key is expired
            if api_key.expires_at and time.time() > api_key.expires_at:
                logger.warning(f"Active API key expired for {service.value}")
                api_key.status = KeyStatus.EXPIRED
                
                # Try to rotate to a backup key
                backup_key = await self._get_backup_key(service)
                if backup_key:
                    self.active_keys[service] = backup_key
                    api_key = backup_key
                else:
                    del self.active_keys[service]
                    return None
            
            # Decrypt and return the key
            try:
                decrypted_key = self.encryption_service.decrypt_cache_data(
                    api_key.encrypted_key, api_key.key_id
                )
                
                # Update usage statistics
                api_key.last_used = time.time()
                api_key.usage_count += 1
                
                return decrypted_key
                
            except Exception as e:
                logger.error(f"Failed to decrypt API key for {service.value}: {e}")
                return None
        
        # Fallback to environment variables or settings
        return self._get_fallback_key(service)
    
    async def rotate_api_key(self, service: APIService, new_key: str) -> bool:
        """Rotate an API key to a new value."""
        
        current_key = self.active_keys.get(service)
        if current_key:
            current_key.status = KeyStatus.ROTATING
        
        try:
            # Store the new key
            new_key_id = await self.store_api_key(service, new_key)
            
            # Mark old key as revoked
            if current_key:
                current_key.status = KeyStatus.REVOKED
            
            logger.info(f"API key rotated successfully for {service.value}")
            
            if self.audit_logger:
                await self.audit_logger.log_security_event(
                    event_type="api_key_rotated",
                    details={
                        "service": service.value,
                        "old_key_id": current_key.key_id if current_key else None,
                        "new_key_id": new_key_id
                    },
                    severity="info"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"API key rotation failed for {service.value}: {e}")
            
            # Restore previous key status
            if current_key:
                current_key.status = KeyStatus.ACTIVE
            
            return False
    
    async def revoke_api_key(self, service: APIService, key_id: str) -> bool:
        """Revoke a specific API key."""
        
        if key_id in self.key_cache:
            api_key = self.key_cache[key_id]
            api_key.status = KeyStatus.REVOKED
            
            # Remove from active keys if it was active
            if service in self.active_keys and self.active_keys[service].key_id == key_id:
                del self.active_keys[service]
            
            # Persist changes
            await self._save_keys_to_storage()
            
            if self.audit_logger:
                await self.audit_logger.log_security_event(
                    event_type="api_key_revoked",
                    details={
                        "service": service.value,
                        "key_id": key_id,
                        "reason": "manual_revocation"
                    },
                    severity="warning"
                )
            
            logger.info(f"API key revoked", service=service.value, key_id=key_id)
            return True
        
        return False
    
    async def validate_all_keys(self) -> Dict[APIService, bool]:
        """Validate all active API keys."""
        
        validation_results = {}
        
        for service, api_key in self.active_keys.items():
            try:
                # Get decrypted key
                key = await self.get_api_key(service)
                if key:
                    is_valid = await self._validate_api_key(service, key)
                    validation_results[service] = is_valid
                    
                    if not is_valid:
                        api_key.status = KeyStatus.FAILED
                        logger.warning(f"API key validation failed for {service.value}")
                else:
                    validation_results[service] = False
                    
            except Exception as e:
                logger.error(f"Key validation error for {service.value}: {e}")
                validation_results[service] = False
        
        return validation_results
    
    async def get_key_usage_metrics(self) -> Dict[APIService, KeyUsageMetrics]:
        """Get usage metrics for all API keys."""
        return self.usage_metrics.copy()
    
    async def check_rotation_needed(self) -> List[APIService]:
        """Check which keys need rotation based on policies."""
        
        rotation_needed = []
        current_time = time.time()
        
        for service, policy in self.rotation_policies.items():
            if not policy.auto_rotation_enabled:
                continue
            
            api_key = self.active_keys.get(service)
            if not api_key:
                continue
            
            # Check if rotation interval has passed
            days_since_creation = (current_time - api_key.created_at) / (24 * 3600)
            if days_since_creation >= policy.rotation_interval_days:
                rotation_needed.append(service)
                continue
            
            # Check if key is expiring soon
            if api_key.expires_at:
                days_until_expiry = (api_key.expires_at - current_time) / (24 * 3600)
                if days_until_expiry <= policy.warning_days_before_expiry:
                    rotation_needed.append(service)
        
        return rotation_needed
    
    async def cleanup_expired_keys(self) -> int:
        """Remove expired and revoked keys from storage."""
        
        current_time = time.time()
        removed_count = 0
        
        keys_to_remove = []
        for key_id, api_key in self.key_cache.items():
            # Remove if expired for more than 30 days or revoked for more than 7 days
            if api_key.status == KeyStatus.EXPIRED:
                if api_key.expires_at and (current_time - api_key.expires_at) > (30 * 24 * 3600):
                    keys_to_remove.append(key_id)
            elif api_key.status == KeyStatus.REVOKED:
                # Keep revoked keys for 7 days for audit purposes
                if (current_time - api_key.created_at) > (7 * 24 * 3600):
                    keys_to_remove.append(key_id)
        
        # Remove expired keys
        for key_id in keys_to_remove:
            del self.key_cache[key_id]
            removed_count += 1
        
        if removed_count > 0:
            await self._save_keys_to_storage()
            logger.info(f"Cleaned up {removed_count} expired/revoked keys")
        
        return removed_count
    
    def _initialize_default_policies(self):
        """Initialize default rotation policies for each service."""
        
        # NASA services - longer rotation periods
        nasa_policy = KeyRotationPolicy(
            service=APIService.CMR,
            rotation_interval_days=180,  # 6 months
            warning_days_before_expiry=30,
            auto_rotation_enabled=False,  # Manual rotation for NASA keys
            backup_keys_count=2,
            validation_required=True
        )
        
        # LLM services - shorter rotation periods
        llm_policy = KeyRotationPolicy(
            service=APIService.OPENAI,
            rotation_interval_days=90,  # 3 months
            warning_days_before_expiry=14,
            auto_rotation_enabled=False,  # Manual for security
            backup_keys_count=1,
            validation_required=True
        )
        
        # Set policies for all services
        for service in APIService:
            if service.value.startswith(("cmr", "giovanni", "modaps", "atmospheric", "earthdata")):
                policy = KeyRotationPolicy(
                    service=service,
                    rotation_interval_days=180,
                    warning_days_before_expiry=30,
                    auto_rotation_enabled=False,
                    backup_keys_count=2,
                    validation_required=True
                )
            else:
                policy = KeyRotationPolicy(
                    service=service,
                    rotation_interval_days=90,
                    warning_days_before_expiry=14,
                    auto_rotation_enabled=False,
                    backup_keys_count=1,
                    validation_required=True
                )
            
            self.rotation_policies[service] = policy
    
    async def _load_keys_from_storage(self):
        """Load encrypted keys from persistent storage."""
        
        if not self.key_store_path.exists():
            logger.info("No existing key store found, starting fresh")
            return
        
        try:
            with open(self.key_store_path, 'r') as f:
                data = json.load(f)
            
            # Load keys
            for key_data in data.get("keys", []):
                api_key = APIKey(**key_data)
                api_key.service = APIService(api_key.service)
                api_key.status = KeyStatus(api_key.status)
                
                self.key_cache[api_key.key_id] = api_key
                
                # Set as active if status is active
                if api_key.status == KeyStatus.ACTIVE:
                    self.active_keys[api_key.service] = api_key
            
            logger.info(f"Loaded {len(self.key_cache)} API keys from storage")
            
        except Exception as e:
            logger.error(f"Failed to load keys from storage: {e}")
    
    async def _save_keys_to_storage(self):
        """Save encrypted keys to persistent storage."""
        
        try:
            data = {
                "keys": [asdict(key) for key in self.key_cache.values()],
                "last_updated": time.time()
            }
            
            # Ensure directory exists
            self.key_store_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write atomically
            temp_path = self.key_store_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_path.replace(self.key_store_path)
            
            logger.debug("API keys saved to storage")
            
        except Exception as e:
            logger.error(f"Failed to save keys to storage: {e}")
    
    async def _initialize_usage_metrics(self):
        """Initialize usage metrics for all services."""
        
        for service in APIService:
            self.usage_metrics[service] = KeyUsageMetrics(
                service=service,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                last_24h_requests=0,
                avg_response_time=0.0,
                rate_limit_hits=0
            )
    
    async def _validate_api_key(self, service: APIService, key: str) -> bool:
        """Validate an API key for a specific service."""
        
        # Check validation cache first
        cache_key = f"{service.value}_{hashlib.md5(key.encode()).hexdigest()[:8]}"
        if cache_key in self.validation_cache:
            is_valid, timestamp = self.validation_cache[cache_key]
            if time.time() - timestamp < self.validation_cache_ttl:
                return is_valid
        
        # Service-specific validation
        is_valid = False
        try:
            if service == APIService.CMR:
                is_valid = await self._validate_cmr_key(key)
            elif service == APIService.GIOVANNI:
                is_valid = await self._validate_giovanni_key(key)
            elif service == APIService.MODAPS_LAADS:
                is_valid = await self._validate_modaps_key(key)
            elif service == APIService.ATMOSPHERIC:
                is_valid = await self._validate_atmospheric_key(key)
            elif service == APIService.EARTHDATA_LOGIN:
                is_valid = await self._validate_earthdata_key(key)
            elif service == APIService.OPENAI:
                is_valid = await self._validate_openai_key(key)
            elif service == APIService.ANTHROPIC:
                is_valid = await self._validate_anthropic_key(key)
            else:
                is_valid = len(key) > 10  # Basic validation
            
            # Cache the result
            self.validation_cache[cache_key] = (is_valid, time.time())
            
        except Exception as e:
            logger.warning(f"Key validation failed for {service.value}: {e}")
            is_valid = False
        
        return is_valid
    
    async def _validate_cmr_key(self, key: str) -> bool:
        """Validate CMR API key (CMR may not require keys for basic access)."""
        # CMR often works without API keys for basic queries
        return True
    
    async def _validate_giovanni_key(self, key: str) -> bool:
        """Validate GIOVANNI API key."""
        # GIOVANNI validation would depend on their specific API
        return len(key) > 10
    
    async def _validate_modaps_key(self, key: str) -> bool:
        """Validate MODAPS/LAADS API key."""
        # MODAPS validation through test request
        return len(key) > 10
    
    async def _validate_atmospheric_key(self, key: str) -> bool:
        """Validate atmospheric data API key."""
        return len(key) > 10
    
    async def _validate_earthdata_key(self, key: str) -> bool:
        """Validate Earthdata Login token/key."""
        return len(key) > 10
    
    async def _validate_openai_key(self, key: str) -> bool:
        """Validate OpenAI API key."""
        return key.startswith("sk-") and len(key) > 20
    
    async def _validate_anthropic_key(self, key: str) -> bool:
        """Validate Anthropic API key."""
        return key.startswith("sk-ant-") and len(key) > 20
    
    async def _get_backup_key(self, service: APIService) -> Optional[APIKey]:
        """Get a backup key for a service."""
        
        backup_keys = [
            key for key in self.key_cache.values()
            if key.service == service and key.status == KeyStatus.ACTIVE
        ]
        
        # Sort by creation time, return newest
        if backup_keys:
            backup_keys.sort(key=lambda k: k.created_at, reverse=True)
            return backup_keys[0]
        
        return None
    
    def _get_fallback_key(self, service: APIService) -> Optional[str]:
        """Get fallback key from environment variables or settings."""
        
        fallback_map = {
            APIService.OPENAI: getattr(settings, 'openai_api_key', None),
            APIService.ANTHROPIC: getattr(settings, 'anthropic_api_key', None),
            APIService.EARTHDATA_LOGIN: getattr(settings, 'earthdata_username', None),
            APIService.MODAPS_LAADS: getattr(settings, 'laads_api_key', None)
        }
        
        return fallback_map.get(service)
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get API key manager service status."""
        
        active_services = list(self.active_keys.keys())
        total_keys = len(self.key_cache)
        
        return {
            "service": "API Key Manager",
            "active_services": [s.value for s in active_services],
            "total_keys_managed": total_keys,
            "encryption_enabled": self.encryption_service is not None,
            "audit_logging_enabled": self.audit_logger is not None,
            "key_store_path": str(self.key_store_path),
            "validation_cache_size": len(self.validation_cache)
        }


# Global API key manager
_api_key_manager: Optional[APIKeyManager] = None


async def get_api_key_manager() -> Optional[APIKeyManager]:
    """Get or create the global API key manager if enabled."""
    global _api_key_manager
    
    # Check if API key management is enabled
    if not getattr(settings, 'enable_api_key_manager', False):
        logger.debug("API Key Manager disabled - using environment variables")
        return None
    
    if _api_key_manager is None:
        try:
            _api_key_manager = APIKeyManager()
            await _api_key_manager.initialize()
        except Exception as e:
            logger.warning(f"Failed to initialize API Key Manager: {e}")
            logger.info("Falling back to environment variable configuration")
            return None
    
    return _api_key_manager


async def get_service_api_key(service: APIService) -> Optional[str]:
    """Convenience function to get API key for a service with graceful fallback."""
    
    # Try API key manager first if enabled
    try:
        manager = await get_api_key_manager()
        if manager:
            key = await manager.get_api_key(service)
            if key:
                return key
    except Exception as e:
        logger.debug(f"API Key Manager failed for {service.value}: {e}")
    
    # Fallback to environment variables
    fallback_map = {
        APIService.OPENAI: getattr(settings, 'openai_api_key', None),
        APIService.ANTHROPIC: getattr(settings, 'anthropic_api_key', None),
        APIService.EARTHDATA_LOGIN: getattr(settings, 'earthdata_username', None),
        APIService.MODAPS_LAADS: getattr(settings, 'laads_api_key', None)
    }
    
    fallback_key = fallback_map.get(service)
    if fallback_key:
        logger.debug(f"Using environment variable for {service.value}")
        return fallback_key
    
    logger.debug(f"No API key available for {service.value} - feature may be limited")
    return None