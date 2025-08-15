"""
NASA Earthdata Login Authentication Service.

Provides OAuth2-based authentication with NASA Earthdata Login for secure access
to NASA Earth science data and services. Supports user authentication, token management,
and session handling.
"""

import asyncio
import aiohttp
import json
import time
import secrets
import hashlib
import base64
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode, parse_qs, urlparse
import structlog

from ..core.config import settings
from ..security.encryption import get_encryption_service

logger = structlog.get_logger(__name__)


class AuthenticationState(Enum):
    """Authentication states."""
    UNAUTHENTICATED = "unauthenticated"
    AUTHENTICATING = "authenticating" 
    AUTHENTICATED = "authenticated"
    TOKEN_EXPIRED = "token_expired"
    REFRESH_NEEDED = "refresh_needed"
    ERROR = "error"


@dataclass
class EarthdataCredentials:
    """Earthdata Login credentials."""
    username: str
    password: str
    client_id: Optional[str] = None
    client_secret: Optional[str] = None


@dataclass
class AuthenticationToken:
    """OAuth2 authentication token."""
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    issued_at: float = None
    
    def __post_init__(self):
        if self.issued_at is None:
            self.issued_at = time.time()
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.expires_in is None:
            return False
        return time.time() > (self.issued_at + self.expires_in - 60)  # 60s buffer
    
    @property
    def expires_at(self) -> datetime:
        """Get token expiration time."""
        return datetime.fromtimestamp(self.issued_at + self.expires_in, tz=timezone.utc)


@dataclass
class UserProfile:
    """Earthdata user profile information."""
    uid: str
    first_name: str
    last_name: str
    email: str
    organization: Optional[str] = None
    country: Optional[str] = None
    user_type: Optional[str] = None
    study_area: Optional[str] = None
    user_groups: List[str] = None
    authorized_apps: List[str] = None
    
    def __post_init__(self):
        if self.user_groups is None:
            self.user_groups = []
        if self.authorized_apps is None:
            self.authorized_apps = []


class EarthdataLoginService:
    """NASA Earthdata Login authentication and user management service."""
    
    def __init__(self):
        # Earthdata Login endpoints
        self.base_url = "https://urs.earthdata.nasa.gov"
        self.auth_url = f"{self.base_url}/oauth/authorize"
        self.token_url = f"{self.base_url}/oauth/token"
        self.profile_url = f"{self.base_url}/api/users"
        self.revoke_url = f"{self.base_url}/oauth/revoke"
        
        # OAuth2 configuration
        self.client_id = getattr(settings, 'earthdata_client_id', None)
        self.client_secret = getattr(settings, 'earthdata_client_secret', None)
        self.redirect_uri = getattr(settings, 'earthdata_redirect_uri', 'http://localhost:8000/auth/callback')
        
        # Check if credentials are available
        self.credentials_available = bool(
            getattr(settings, 'earthdata_username', None) and 
            getattr(settings, 'earthdata_password', None)
        )
        
        # Session management
        self.session = None
        self.current_token = None
        self.user_profile = None
        self.auth_state = AuthenticationState.UNAUTHENTICATED
        
        # Security
        self.encryption_service = None
        self.active_states = {}  # PKCE state tracking
        
        # Token storage (in production, use secure storage)
        self.token_cache = {}
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def initialize(self):
        """Initialize the Earthdata authentication service."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": "NASA-CMR-Agent/1.0",
                    "Accept": "application/json"
                }
            )
        
        # Initialize encryption service for secure token storage
        try:
            self.encryption_service = get_encryption_service()
        except Exception as e:
            logger.debug(f"Encryption service not available: {e}")
        
        # Log initialization status
        if self.credentials_available:
            logger.info("Earthdata Login service initialized with credentials")
        else:
            logger.info("Earthdata Login service initialized without credentials - enhanced features disabled")
    
    async def close(self):
        """Close the authentication service."""
        if self.session:
            await self.session.close()
            self.session = None
    
    def generate_authorization_url(self, state: Optional[str] = None,
                                 scopes: List[str] = None) -> Tuple[str, str]:
        """Generate OAuth2 authorization URL with PKCE."""
        
        if state is None:
            state = secrets.token_urlsafe(32)
        
        # Generate PKCE code challenge
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).decode().rstrip('=')
        
        # Store state and code verifier
        self.active_states[state] = {
            "code_verifier": code_verifier,
            "created_at": time.time(),
            "scopes": scopes or []
        }
        
        # Build authorization URL
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256"
        }
        
        if scopes:
            params["scope"] = " ".join(scopes)
        
        auth_url = f"{self.auth_url}?{urlencode(params)}"
        
        logger.info("Generated Earthdata authorization URL", state=state)
        return auth_url, state
    
    async def handle_oauth_callback(self, authorization_code: str, 
                                  state: str) -> AuthenticationToken:
        """Handle OAuth2 callback and exchange code for token."""
        
        if not self.session:
            await self.initialize()
        
        # Validate state
        if state not in self.active_states:
            raise ValueError("Invalid or expired state parameter")
        
        state_data = self.active_states.pop(state)
        code_verifier = state_data["code_verifier"]
        
        # Exchange authorization code for access token
        token_data = {
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "code_verifier": code_verifier
        }
        
        # Add client secret if available
        if self.client_secret:
            token_data["client_secret"] = self.client_secret
        
        try:
            async with self.session.post(self.token_url, data=token_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"Token exchange failed: {error_text}"
                    )
                
                token_response = await response.json()
                
                # Create authentication token
                self.current_token = AuthenticationToken(
                    access_token=token_response["access_token"],
                    token_type=token_response.get("token_type", "Bearer"),
                    expires_in=token_response.get("expires_in", 3600),
                    refresh_token=token_response.get("refresh_token"),
                    scope=token_response.get("scope")
                )
                
                self.auth_state = AuthenticationState.AUTHENTICATED
                
                # Securely store token
                await self._store_token_securely(self.current_token)
                
                logger.info("Successfully authenticated with Earthdata Login")
                return self.current_token
                
        except Exception as e:
            self.auth_state = AuthenticationState.ERROR
            logger.error(f"OAuth callback handling failed: {e}")
            raise
    
    async def authenticate_with_credentials(self, credentials: EarthdataCredentials) -> AuthenticationToken:
        """Authenticate using username/password (Resource Owner Password Credentials flow)."""
        
        if not self.session:
            await self.initialize()
        
        self.auth_state = AuthenticationState.AUTHENTICATING
        
        # Use client credentials if provided, otherwise use username/password flow
        if credentials.client_id and credentials.client_secret:
            token_data = {
                "grant_type": "client_credentials",
                "client_id": credentials.client_id,
                "client_secret": credentials.client_secret
            }
        else:
            token_data = {
                "grant_type": "password",
                "username": credentials.username,
                "password": credentials.password,
                "client_id": self.client_id
            }
        
        try:
            async with self.session.post(self.token_url, data=token_data) as response:
                if response.status != 200:
                    self.auth_state = AuthenticationState.ERROR
                    error_text = await response.text()
                    
                    # Handle common error cases
                    if response.status == 401:
                        raise ValueError("Invalid credentials")
                    elif response.status == 400:
                        raise ValueError("Invalid request parameters")
                    else:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"Authentication failed: {error_text}"
                        )
                
                token_response = await response.json()
                
                self.current_token = AuthenticationToken(
                    access_token=token_response["access_token"],
                    token_type=token_response.get("token_type", "Bearer"),
                    expires_in=token_response.get("expires_in", 3600),
                    refresh_token=token_response.get("refresh_token"),
                    scope=token_response.get("scope")
                )
                
                self.auth_state = AuthenticationState.AUTHENTICATED
                
                # Store token securely
                await self._store_token_securely(self.current_token)
                
                logger.info("Successfully authenticated with credentials")
                return self.current_token
                
        except Exception as e:
            self.auth_state = AuthenticationState.ERROR
            logger.error(f"Credential authentication failed: {e}")
            raise
    
    async def refresh_token(self) -> AuthenticationToken:
        """Refresh the current access token."""
        
        if not self.current_token or not self.current_token.refresh_token:
            raise ValueError("No refresh token available")
        
        if not self.session:
            await self.initialize()
        
        self.auth_state = AuthenticationState.REFRESH_NEEDED
        
        token_data = {
            "grant_type": "refresh_token",
            "refresh_token": self.current_token.refresh_token,
            "client_id": self.client_id
        }
        
        if self.client_secret:
            token_data["client_secret"] = self.client_secret
        
        try:
            async with self.session.post(self.token_url, data=token_data) as response:
                if response.status != 200:
                    self.auth_state = AuthenticationState.TOKEN_EXPIRED
                    error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"Token refresh failed: {error_text}"
                    )
                
                token_response = await response.json()
                
                # Update current token
                self.current_token = AuthenticationToken(
                    access_token=token_response["access_token"],
                    token_type=token_response.get("token_type", "Bearer"),
                    expires_in=token_response.get("expires_in", 3600),
                    refresh_token=token_response.get("refresh_token", self.current_token.refresh_token),
                    scope=token_response.get("scope")
                )
                
                self.auth_state = AuthenticationState.AUTHENTICATED
                
                # Store refreshed token
                await self._store_token_securely(self.current_token)
                
                logger.info("Successfully refreshed authentication token")
                return self.current_token
                
        except Exception as e:
            self.auth_state = AuthenticationState.ERROR
            logger.error(f"Token refresh failed: {e}")
            raise
    
    async def get_user_profile(self) -> UserProfile:
        """Get authenticated user's profile information."""
        
        if not self.is_authenticated():
            raise ValueError("User not authenticated")
        
        if not self.session:
            await self.initialize()
        
        # Check if token needs refresh
        if self.current_token.is_expired:
            await self.refresh_token()
        
        headers = {"Authorization": f"{self.current_token.token_type} {self.current_token.access_token}"}
        
        try:
            async with self.session.get(f"{self.profile_url}/user", headers=headers) as response:
                if response.status == 401:
                    # Token might be invalid, try refresh
                    await self.refresh_token()
                    headers["Authorization"] = f"{self.current_token.token_type} {self.current_token.access_token}"
                    
                    async with self.session.get(f"{self.profile_url}/user", headers=headers) as retry_response:
                        if retry_response.status != 200:
                            raise aiohttp.ClientResponseError(
                                request_info=retry_response.request_info,
                                history=retry_response.history,
                                status=retry_response.status,
                                message="Failed to get user profile after token refresh"
                            )
                        profile_data = await retry_response.json()
                
                elif response.status != 200:
                    error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"Profile request failed: {error_text}"
                    )
                else:
                    profile_data = await response.json()
                
                # Parse user profile
                self.user_profile = UserProfile(
                    uid=profile_data["uid"],
                    first_name=profile_data.get("first_name", ""),
                    last_name=profile_data.get("last_name", ""),
                    email=profile_data.get("email", ""),
                    organization=profile_data.get("organization"),
                    country=profile_data.get("country"),
                    user_type=profile_data.get("user_type"),
                    study_area=profile_data.get("study_area"),
                    user_groups=profile_data.get("user_groups", []),
                    authorized_apps=profile_data.get("authorized_apps", [])
                )
                
                logger.info("Retrieved user profile", uid=self.user_profile.uid)
                return self.user_profile
                
        except Exception as e:
            logger.error(f"Failed to get user profile: {e}")
            raise
    
    async def logout(self):
        """Logout and revoke current token."""
        
        if self.current_token and self.session:
            # Revoke token
            try:
                token_data = {
                    "token": self.current_token.access_token,
                    "client_id": self.client_id
                }
                
                if self.client_secret:
                    token_data["client_secret"] = self.client_secret
                
                async with self.session.post(self.revoke_url, data=token_data) as response:
                    if response.status not in [200, 204]:
                        logger.warning(f"Token revocation failed: {response.status}")
                    else:
                        logger.info("Token successfully revoked")
                        
            except Exception as e:
                logger.warning(f"Token revocation error: {e}")
        
        # Clear authentication state
        self.current_token = None
        self.user_profile = None
        self.auth_state = AuthenticationState.UNAUTHENTICATED
        
        # Clear secure storage
        await self._clear_stored_token()
    
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated."""
        return (self.auth_state == AuthenticationState.AUTHENTICATED and 
                self.current_token is not None and 
                not self.current_token.is_expired)
    
    async def ensure_authenticated(self) -> AuthenticationToken:
        """Ensure user is authenticated, refresh token if needed."""
        
        if not self.current_token:
            # Try to load from secure storage
            await self._load_stored_token()
        
        if not self.current_token:
            raise ValueError("No authentication token available")
        
        if self.current_token.is_expired:
            if self.current_token.refresh_token:
                return await self.refresh_token()
            else:
                raise ValueError("Token expired and no refresh token available")
        
        return self.current_token
    
    def get_authorization_header(self) -> Dict[str, str]:
        """Get authorization header for API requests."""
        
        if not self.is_authenticated():
            raise ValueError("User not authenticated")
        
        return {
            "Authorization": f"{self.current_token.token_type} {self.current_token.access_token}"
        }
    
    async def _store_token_securely(self, token: AuthenticationToken):
        """Store authentication token securely."""
        
        if self.encryption_service:
            try:
                # Encrypt token data
                token_data = asdict(token)
                encrypted_data, key_id = self.encryption_service.encrypt_cache_data(token_data)
                
                # Store encrypted token (in production, use secure storage like keychain)
                self.token_cache["encrypted_token"] = encrypted_data
                self.token_cache["key_id"] = key_id
                self.token_cache["stored_at"] = time.time()
                
                logger.debug("Token stored securely")
            except Exception as e:
                logger.warning(f"Failed to store token securely: {e}")
        else:
            logger.warning("Encryption service not available - token stored in memory only")
    
    async def _load_stored_token(self):
        """Load stored authentication token."""
        
        if not self.token_cache.get("encrypted_token"):
            return
        
        if self.encryption_service:
            try:
                # Decrypt token data
                decrypted_data = self.encryption_service.decrypt_cache_data(
                    self.token_cache["encrypted_token"],
                    self.token_cache["key_id"]
                )
                
                # Recreate token object
                self.current_token = AuthenticationToken(**decrypted_data)
                
                # Check if token is still valid
                if not self.current_token.is_expired:
                    self.auth_state = AuthenticationState.AUTHENTICATED
                    logger.debug("Loaded stored authentication token")
                else:
                    logger.debug("Stored token has expired")
                    self.current_token = None
                    
            except Exception as e:
                logger.warning(f"Failed to load stored token: {e}")
                await self._clear_stored_token()
    
    async def _clear_stored_token(self):
        """Clear stored authentication token."""
        self.token_cache.clear()
        logger.debug("Cleared stored authentication token")
    
    def get_auth_status(self) -> Dict[str, Any]:
        """Get current authentication status."""
        
        return {
            "authenticated": self.is_authenticated(),
            "auth_state": self.auth_state.value,
            "user_id": self.user_profile.uid if self.user_profile else None,
            "token_expires_at": self.current_token.expires_at.isoformat() if self.current_token else None,
            "has_refresh_token": bool(self.current_token and self.current_token.refresh_token),
            "active_oauth_states": len(self.active_states)
        }


# Global Earthdata authentication service
_earthdata_auth_service: Optional[EarthdataLoginService] = None


async def get_earthdata_auth_service() -> EarthdataLoginService:
    """Get or create the global Earthdata authentication service."""
    global _earthdata_auth_service
    
    if _earthdata_auth_service is None:
        _earthdata_auth_service = EarthdataLoginService()
        await _earthdata_auth_service.initialize()
    
    return _earthdata_auth_service


async def close_earthdata_auth_service():
    """Close the global Earthdata authentication service."""
    global _earthdata_auth_service
    
    if _earthdata_auth_service:
        await _earthdata_auth_service.close()
        _earthdata_auth_service = None