"""
Enterprise-Grade Encryption System for NASA CMR Agent.

Provides AES-256 encryption, key management, and secure data protection
for sensitive information including query data, cache contents, and logs.
"""

import os
import base64
import hashlib
import secrets
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone, timedelta
import structlog

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

logger = structlog.get_logger(__name__)


class EncryptionMethod(Enum):
    """Available encryption methods."""
    FERNET = "fernet"
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"


class KeyType(Enum):
    """Types of encryption keys."""
    MASTER = "master"
    DATA = "data"
    CACHE = "cache"
    LOG = "log"
    TEMP = "temp"


@dataclass
class EncryptionResult:
    """Encryption operation result."""
    success: bool
    encrypted_data: Optional[bytes] = None
    error_message: Optional[str] = None
    key_id: Optional[str] = None
    method: Optional[EncryptionMethod] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class DecryptionResult:
    """Decryption operation result."""
    success: bool
    decrypted_data: Optional[Union[str, bytes]] = None
    error_message: Optional[str] = None
    key_id: Optional[str] = None
    method: Optional[EncryptionMethod] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class EncryptionKey:
    """Encryption key metadata."""
    key_id: str
    key_type: KeyType
    method: EncryptionMethod
    created_at: str
    expires_at: Optional[str] = None
    is_active: bool = True
    rotation_count: int = 0


class SecurityEncryptionService:
    """Enterprise-grade encryption service."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography package required for encryption features")
        
        self.encryption_stats = {
            "total_encryptions": 0,
            "total_decryptions": 0,
            "failed_operations": 0,
            "keys_generated": 0,
            "keys_rotated": 0
        }
        
        # Key storage (in production, use secure key management service)
        self._keys: Dict[str, Dict[str, Any]] = {}
        self._key_metadata: Dict[str, EncryptionKey] = {}
        
        # Initialize master key
        if master_key:
            self._master_key = master_key
        else:
            self._master_key = self._generate_master_key()
        
        # Initialize default keys
        self._initialize_default_keys()
    
    def _generate_master_key(self) -> bytes:
        """Generate a secure master key."""
        return secrets.token_bytes(32)  # 256-bit key
    
    def _initialize_default_keys(self):
        """Initialize default encryption keys."""
        default_key_types = [
            (KeyType.DATA, EncryptionMethod.FERNET),
            (KeyType.CACHE, EncryptionMethod.AES_256_GCM),
            (KeyType.LOG, EncryptionMethod.FERNET),
        ]
        
        for key_type, method in default_key_types:
            self.generate_key(key_type, method)
    
    def generate_key(self, key_type: KeyType, method: EncryptionMethod, 
                    expires_hours: Optional[int] = None) -> str:
        """Generate a new encryption key."""
        key_id = f"{key_type.value}_{secrets.token_hex(8)}"
        
        if method == EncryptionMethod.FERNET:
            key_material = Fernet.generate_key()
            cipher_key = key_material
        elif method in [EncryptionMethod.AES_256_GCM, EncryptionMethod.AES_256_CBC]:
            key_material = secrets.token_bytes(32)  # 256-bit key
            cipher_key = key_material
        else:
            raise ValueError(f"Unsupported encryption method: {method}")
        
        # Calculate expiration
        expires_at = None
        if expires_hours:
            expires_at = (datetime.now(timezone.utc) + 
                         timedelta(hours=expires_hours)).isoformat()
        
        # Store key and metadata
        self._keys[key_id] = {
            "key_material": key_material,
            "cipher_key": cipher_key,
            "method": method
        }
        
        self._key_metadata[key_id] = EncryptionKey(
            key_id=key_id,
            key_type=key_type,
            method=method,
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at=expires_at
        )
        
        self.encryption_stats["keys_generated"] += 1
        logger.info(f"Generated {method.value} key for {key_type.value}: {key_id}")
        
        return key_id
    
    def encrypt_data(self, data: Union[str, bytes], key_type: KeyType = KeyType.DATA) -> EncryptionResult:
        """Encrypt data using specified key type."""
        self.encryption_stats["total_encryptions"] += 1
        
        try:
            # Get active key for the type
            key_id = self._get_active_key_id(key_type)
            if not key_id:
                return EncryptionResult(
                    success=False,
                    error_message=f"No active key found for type {key_type.value}"
                )
            
            key_info = self._keys[key_id]
            method = key_info["method"]
            
            # Convert string to bytes
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Encrypt based on method
            if method == EncryptionMethod.FERNET:
                encrypted_data = self._encrypt_fernet(data_bytes, key_info["cipher_key"])
            elif method == EncryptionMethod.AES_256_GCM:
                encrypted_data = self._encrypt_aes_gcm(data_bytes, key_info["cipher_key"])
            elif method == EncryptionMethod.AES_256_CBC:
                encrypted_data = self._encrypt_aes_cbc(data_bytes, key_info["cipher_key"])
            else:
                return EncryptionResult(
                    success=False,
                    error_message=f"Unsupported encryption method: {method}"
                )
            
            return EncryptionResult(
                success=True,
                encrypted_data=encrypted_data,
                key_id=key_id,
                method=method
            )
            
        except Exception as e:
            self.encryption_stats["failed_operations"] += 1
            logger.error(f"Encryption failed: {e}")
            return EncryptionResult(
                success=False,
                error_message=str(e)
            )
    
    def decrypt_data(self, encrypted_data: bytes, key_id: str) -> DecryptionResult:
        """Decrypt data using specified key."""
        self.encryption_stats["total_decryptions"] += 1
        
        try:
            if key_id not in self._keys:
                return DecryptionResult(
                    success=False,
                    error_message=f"Key not found: {key_id}"
                )
            
            key_info = self._keys[key_id]
            method = key_info["method"]
            
            # Check if key is expired
            if self._is_key_expired(key_id):
                logger.warning(f"Attempting to use expired key: {key_id}")
                return DecryptionResult(
                    success=False,
                    error_message=f"Key expired: {key_id}"
                )
            
            # Decrypt based on method
            if method == EncryptionMethod.FERNET:
                decrypted_data = self._decrypt_fernet(encrypted_data, key_info["cipher_key"])
            elif method == EncryptionMethod.AES_256_GCM:
                decrypted_data = self._decrypt_aes_gcm(encrypted_data, key_info["cipher_key"])
            elif method == EncryptionMethod.AES_256_CBC:
                decrypted_data = self._decrypt_aes_cbc(encrypted_data, key_info["cipher_key"])
            else:
                return DecryptionResult(
                    success=False,
                    error_message=f"Unsupported decryption method: {method}"
                )
            
            return DecryptionResult(
                success=True,
                decrypted_data=decrypted_data.decode('utf-8'),
                key_id=key_id,
                method=method
            )
            
        except Exception as e:
            self.encryption_stats["failed_operations"] += 1
            logger.error(f"Decryption failed: {e}")
            return DecryptionResult(
                success=False,
                error_message=str(e)
            )
    
    def encrypt_sensitive_query(self, query: str) -> Tuple[bytes, str]:
        """Encrypt a NASA CMR query for secure storage."""
        result = self.encrypt_data(query, KeyType.DATA)
        if not result.success:
            raise ValueError(f"Query encryption failed: {result.error_message}")
        return result.encrypted_data, result.key_id
    
    def decrypt_sensitive_query(self, encrypted_query: bytes, key_id: str) -> str:
        """Decrypt a NASA CMR query from secure storage."""
        result = self.decrypt_data(encrypted_query, key_id)
        if not result.success:
            raise ValueError(f"Query decryption failed: {result.error_message}")
        return result.decrypted_data
    
    def encrypt_cache_data(self, cache_data: Dict[str, Any]) -> Tuple[bytes, str]:
        """Encrypt cache data for secure storage."""
        import json
        json_data = json.dumps(cache_data)
        result = self.encrypt_data(json_data, KeyType.CACHE)
        if not result.success:
            raise ValueError(f"Cache encryption failed: {result.error_message}")
        return result.encrypted_data, result.key_id
    
    def decrypt_cache_data(self, encrypted_cache: bytes, key_id: str) -> Dict[str, Any]:
        """Decrypt cache data from secure storage."""
        import json
        result = self.decrypt_data(encrypted_cache, key_id)
        if not result.success:
            raise ValueError(f"Cache decryption failed: {result.error_message}")
        return json.loads(result.decrypted_data)
    
    def rotate_key(self, key_type: KeyType) -> str:
        """Rotate encryption key for specified type."""
        # Get current active key
        old_key_id = self._get_active_key_id(key_type)
        if old_key_id:
            # Deactivate old key
            self._key_metadata[old_key_id].is_active = False
            self._key_metadata[old_key_id].rotation_count += 1
        
        # Generate new key
        method = self._key_metadata[old_key_id].method if old_key_id else EncryptionMethod.FERNET
        new_key_id = self.generate_key(key_type, method)
        
        self.encryption_stats["keys_rotated"] += 1
        logger.info(f"Rotated key for {key_type.value}: {old_key_id} -> {new_key_id}")
        
        return new_key_id
    
    def _encrypt_fernet(self, data: bytes, key: bytes) -> bytes:
        """Encrypt data using Fernet."""
        f = Fernet(key)
        return f.encrypt(data)
    
    def _decrypt_fernet(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt data using Fernet."""
        f = Fernet(key)
        return f.decrypt(encrypted_data)
    
    def _encrypt_aes_gcm(self, data: bytes, key: bytes) -> bytes:
        """Encrypt data using AES-256-GCM."""
        # Generate random nonce
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        
        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
        encryptor = cipher.encryptor()
        
        # Encrypt data
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Return nonce + tag + ciphertext
        return nonce + encryptor.tag + ciphertext
    
    def _decrypt_aes_gcm(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt data using AES-256-GCM."""
        # Extract components
        nonce = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]
        
        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        
        # Decrypt data
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def _encrypt_aes_cbc(self, data: bytes, key: bytes) -> bytes:
        """Encrypt data using AES-256-CBC."""
        # Generate random IV
        iv = secrets.token_bytes(16)  # 128-bit IV
        
        # Pad data to block size
        padded_data = self._pad_data(data)
        
        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        # Encrypt data
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        # Return IV + ciphertext
        return iv + ciphertext
    
    def _decrypt_aes_cbc(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt data using AES-256-CBC."""
        # Extract IV and ciphertext
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        
        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        
        # Decrypt data
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        return self._unpad_data(padded_data)
    
    def _pad_data(self, data: bytes) -> bytes:
        """Apply PKCS7 padding."""
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _unpad_data(self, padded_data: bytes) -> bytes:
        """Remove PKCS7 padding."""
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def _get_active_key_id(self, key_type: KeyType) -> Optional[str]:
        """Get active key ID for specified type."""
        for key_id, metadata in self._key_metadata.items():
            if (metadata.key_type == key_type and 
                metadata.is_active and 
                not self._is_key_expired(key_id)):
                return key_id
        return None
    
    def _is_key_expired(self, key_id: str) -> bool:
        """Check if key is expired."""
        metadata = self._key_metadata.get(key_id)
        if not metadata or not metadata.expires_at:
            return False
        
        expires_at = datetime.fromisoformat(metadata.expires_at)
        return datetime.now(timezone.utc) > expires_at
    
    def get_key_info(self, key_id: str) -> Optional[EncryptionKey]:
        """Get key metadata."""
        return self._key_metadata.get(key_id)
    
    def list_keys(self, key_type: Optional[KeyType] = None, 
                 active_only: bool = False) -> Dict[str, EncryptionKey]:
        """List encryption keys."""
        filtered_keys = {}
        
        for key_id, metadata in self._key_metadata.items():
            # Filter by type if specified
            if key_type and metadata.key_type != key_type:
                continue
            
            # Filter by active status if specified
            if active_only and (not metadata.is_active or self._is_key_expired(key_id)):
                continue
            
            filtered_keys[key_id] = metadata
        
        return filtered_keys
    
    def cleanup_expired_keys(self) -> int:
        """Remove expired keys and return count."""
        expired_keys = []
        
        for key_id, metadata in self._key_metadata.items():
            if self._is_key_expired(key_id):
                expired_keys.append(key_id)
        
        for key_id in expired_keys:
            del self._keys[key_id]
            del self._key_metadata[key_id]
            logger.info(f"Cleaned up expired key: {key_id}")
        
        return len(expired_keys)
    
    def get_encryption_stats(self) -> Dict[str, Any]:
        """Get encryption service statistics."""
        active_keys = sum(1 for metadata in self._key_metadata.values() 
                         if metadata.is_active and not self._is_key_expired(metadata.key_id))
        
        return {
            **self.encryption_stats,
            "active_keys": active_keys,
            "total_keys": len(self._key_metadata),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Global encryption service instance
_encryption_service: Optional[SecurityEncryptionService] = None


def get_encryption_service(master_key: Optional[bytes] = None) -> SecurityEncryptionService:
    """Get or create the global encryption service."""
    global _encryption_service
    
    if _encryption_service is None:
        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning("Cryptography package not available - encryption disabled")
            return None
        
        _encryption_service = SecurityEncryptionService(master_key)
    
    return _encryption_service


def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token."""
    return secrets.token_urlsafe(length)


def hash_data(data: Union[str, bytes], salt: Optional[bytes] = None) -> Tuple[str, bytes]:
    """Hash data with optional salt using SHA-256."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    if salt is None:
        salt = secrets.token_bytes(32)
    
    hash_obj = hashlib.sha256(data + salt)
    return hash_obj.hexdigest(), salt