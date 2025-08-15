from typing import Optional
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY")
    openai_model: str = Field("gpt-4-turbo-preview", env="OPENAI_MODEL")
    anthropic_model: str = Field("claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")
    gemini_model: str = Field("gemini-1.5-pro", env="GEMINI_MODEL")
    
    # NASA CMR API Configuration
    cmr_base_url: str = Field("https://cmr.earthdata.nasa.gov/search", env="CMR_BASE_URL")
    cmr_request_timeout: int = Field(30, env="CMR_REQUEST_TIMEOUT")
    cmr_max_retries: int = Field(3, env="CMR_MAX_RETRIES")
    cmr_rate_limit_per_second: int = Field(10, env="CMR_RATE_LIMIT_PER_SECOND")
    
    # Redis Configuration
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    redis_db: int = Field(0, env="REDIS_DB")
    redis_cache_ttl: int = Field(3600, env="REDIS_CACHE_TTL")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_debug: bool = Field(False, env="API_DEBUG")
    api_reload: bool = Field(False, env="API_RELOAD")
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")
    
    # Performance Configuration
    max_concurrent_requests: int = Field(10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(60, env="REQUEST_TIMEOUT")
    circuit_breaker_failure_threshold: int = Field(5, env="CIRCUIT_BREAKER_FAILURE_THRESHOLD")
    circuit_breaker_recovery_timeout: int = Field(30, env="CIRCUIT_BREAKER_RECOVERY_TIMEOUT")
    
    # Monitoring
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    
    # Development
    environment: str = Field("development", env="ENVIRONMENT")
    
    # Vector Database Configuration (Weaviate)
    weaviate_url: str = Field("http://localhost:8080", env="WEAVIATE_URL")
    weaviate_api_key: Optional[str] = Field(None, env="WEAVIATE_API_KEY")
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # Knowledge Graph Configuration (Neo4j)
    neo4j_uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field("neo4j", env="NEO4J_USER")
    neo4j_password: str = Field("password", env="NEO4J_PASSWORD")
    neo4j_database: str = Field("neo4j", env="NEO4J_DATABASE")
    
    # Advanced Features
    enable_vector_search: bool = Field(True, env="ENABLE_VECTOR_SEARCH")
    enable_knowledge_graph: bool = Field(True, env="ENABLE_KNOWLEDGE_GRAPH")
    enable_rag: bool = Field(True, env="ENABLE_RAG")
    
    # API Key Management (Optional - for production deployments)
    enable_api_key_manager: bool = Field(False, env="ENABLE_API_KEY_MANAGER")
    api_key_encryption_key: Optional[str] = Field(None, env="API_KEY_ENCRYPTION_KEY")
    
    # NASA API Credentials (Optional - for enhanced features)
    earthdata_username: Optional[str] = Field(None, env="EARTHDATA_USERNAME")
    earthdata_password: Optional[str] = Field(None, env="EARTHDATA_PASSWORD")
    earthdata_client_id: Optional[str] = Field(None, env="EARTHDATA_CLIENT_ID")
    earthdata_client_secret: Optional[str] = Field(None, env="EARTHDATA_CLIENT_SECRET")
    earthdata_redirect_uri: Optional[str] = Field("http://localhost:8000/auth/callback", env="EARTHDATA_REDIRECT_URI")
    laads_api_key: Optional[str] = Field(None, env="LAADS_API_KEY")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


settings = Settings()