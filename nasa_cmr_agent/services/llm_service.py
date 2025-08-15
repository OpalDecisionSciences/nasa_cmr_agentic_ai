from typing import Optional, Dict, Any, Union, List
import asyncio
import structlog
import httpx
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.language_models.base import BaseLanguageModel
from tenacity import retry, stop_after_attempt, wait_exponential

# Import LLM providers (with fallback handling for missing packages)
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from langchain_community.llms import HuggingFaceHub
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

from ..core.config import settings

logger = structlog.get_logger(__name__)


class LLMService:
    """
    Multi-provider LLM service with intelligent fallback mechanisms.
    
    Supports OpenAI, Anthropic, Gemini, DeepSeek, Cohere, and Together
    with automatic fallback, rate limiting, and error handling.
    """
    
    def __init__(self):
        self.providers = self._initialize_all_providers()
        self.current_provider_index = 0
        self.failed_providers = set()
    
    def _initialize_all_providers(self) -> List[Dict[str, Any]]:
        """Initialize all available LLM providers in priority order."""
        providers = []
        
        for provider_name in settings.llm_provider_priority:
            provider_info = self._initialize_provider(provider_name)
            if provider_info:
                providers.append(provider_info)
                logger.info(f"Initialized LLM provider", provider=provider_name, model=provider_info["model"])
        
        if not providers:
            logger.error("No LLM providers available!")
        else:
            logger.info(f"LLM service ready", total_providers=len(providers), primary=providers[0]["name"])
            
        return providers
    
    def _initialize_provider(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Initialize a specific LLM provider with comprehensive support."""
        try:
            # OpenAI (Commercial)
            if provider_name == "openai" and OPENAI_AVAILABLE and settings.openai_api_key:
                return {
                    "name": "openai",
                    "model": settings.openai_model,
                    "type": "commercial",
                    "client": ChatOpenAI(
                        api_key=settings.openai_api_key,
                        model=settings.openai_model,
                        temperature=0.1,
                        max_tokens=1000,
                        timeout=30
                    )
                }
            
            # Anthropic
            elif provider_name == "anthropic" and ANTHROPIC_AVAILABLE and settings.anthropic_api_key:
                return {
                    "name": "anthropic",
                    "model": settings.anthropic_model,
                    "type": "commercial",
                    "client": ChatAnthropic(
                        api_key=settings.anthropic_api_key,
                        model=settings.anthropic_model,
                        temperature=0.1,
                        max_tokens=1000,
                        timeout=30
                    )
                }
            
            # Google Gemini
            elif provider_name == "gemini" and GEMINI_AVAILABLE and settings.gemini_api_key:
                return {
                    "name": "gemini",
                    "model": settings.gemini_model,
                    "type": "commercial",
                    "client": ChatGoogleGenerativeAI(
                        google_api_key=settings.gemini_api_key,
                        model=settings.gemini_model,
                        temperature=0.1,
                        max_tokens=1000,
                        timeout=30
                    )
                }
            
            # DeepSeek (OpenAI-compatible)
            elif provider_name == "deepseek" and OPENAI_AVAILABLE and settings.deepseek_api_key:
                return {
                    "name": "deepseek",
                    "model": settings.deepseek_model,
                    "type": "alternative",
                    "client": ChatOpenAI(
                        api_key=settings.deepseek_api_key,
                        model=settings.deepseek_model,
                        base_url="https://api.deepseek.com/v1",
                        temperature=0.1,
                        max_tokens=1000,
                        timeout=30
                    )
                }
            
            # HuggingFace
            elif provider_name == "huggingface" and HUGGINGFACE_AVAILABLE and settings.huggingface_api_key:
                return {
                    "name": "huggingface",
                    "model": settings.huggingface_model,
                    "type": "opensource",
                    "client": HuggingFaceHub(
                        huggingfacehub_api_token=settings.huggingface_api_key,
                        repo_id=settings.huggingface_model,
                        model_kwargs={"temperature": 0.1, "max_length": 1000}
                    )
                }
            
            # Meta (OpenAI-compatible endpoint)
            elif provider_name == "meta" and OPENAI_AVAILABLE and settings.meta_api_key:
                return {
                    "name": "meta",
                    "model": settings.meta_model,
                    "type": "opensource",
                    "client": ChatOpenAI(
                        api_key=settings.meta_api_key,
                        model=settings.meta_model,
                        base_url=settings.meta_base_url,
                        temperature=0.1,
                        max_tokens=1000,
                        timeout=30
                    )
                }
            
            # OpenAI Open Source (newer models)
            elif provider_name == "openai_opensource" and OPENAI_AVAILABLE and settings.openai_opensource_api_key:
                return {
                    "name": "openai_opensource",
                    "model": settings.openai_opensource_model,
                    "type": "opensource",
                    "client": ChatOpenAI(
                        api_key=settings.openai_opensource_api_key,
                        model=settings.openai_opensource_model,
                        base_url=settings.openai_opensource_base_url,
                        temperature=0.1,
                        max_tokens=1000,
                        timeout=30
                    )
                }
            
            # Together AI (OpenAI-compatible)
            elif provider_name == "together" and OPENAI_AVAILABLE and settings.together_api_key:
                return {
                    "name": "together",
                    "model": settings.together_model,
                    "type": "opensource",
                    "client": ChatOpenAI(
                        api_key=settings.together_api_key,
                        model=settings.together_model,
                        base_url="https://api.together.xyz/v1",
                        temperature=0.1,
                        max_tokens=1000,
                        timeout=30
                    )
                }
            
            # Cohere (requires additional package)
            elif provider_name == "cohere" and settings.cohere_api_key:
                logger.warning("Cohere integration requires langchain-cohere package - skipping")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to initialize {provider_name} LLM", error=str(e))
        
        return None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate(
        self, 
        prompt: Union[str, BaseMessage], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate response using LLM with intelligent fallback mechanism.
        
        Args:
            prompt: Input prompt or message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
            
        Raises:
            Exception: If all available LLM providers fail
        """
        if not self.providers:
            raise Exception("No LLM providers available")
        
        # Try providers in priority order, skipping failed ones
        for i, provider in enumerate(self.providers):
            provider_name = provider["name"]
            
            # Skip providers that have recently failed
            if provider_name in self.failed_providers:
                continue
                
            try:
                logger.debug(f"Attempting generation with {provider_name}")
                
                response = await self._generate_with_provider(
                    provider, prompt, max_tokens, temperature, **kwargs
                )
                
                # Success - update current provider and remove from failed set
                self.current_provider_index = i
                self.failed_providers.discard(provider_name)
                
                logger.info(f"Generation successful", provider=provider_name, model=provider["model"])
                return response
                
            except Exception as e:
                logger.warning(f"Generation failed with {provider_name}", error=str(e))
                self.failed_providers.add(provider_name)
                continue
        
        # If we get here, all providers failed
        logger.error("All LLM providers failed")
        
        # Reset failed providers for next attempt
        self.failed_providers.clear()
        raise Exception("All available LLM providers failed")
    
    async def _generate_with_provider(
        self,
        provider: Dict[str, Any],
        prompt: Union[str, BaseMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate response with specific provider."""
        llm = provider["client"]
        
        # Update LLM parameters if provided
        if max_tokens is not None:
            llm.max_tokens = max_tokens
        if temperature is not None:
            llm.temperature = temperature
        
        # Convert string prompt to message if needed
        if isinstance(prompt, str):
            messages = [HumanMessage(content=prompt)]
        else:
            messages = [prompt]
        
        # Generate response
        response = await llm.ainvoke(messages, **kwargs)
        return response.content
    
    async def batch_generate(
        self,
        prompts: list[Union[str, BaseMessage]],
        max_concurrent: int = 5,
        **kwargs
    ) -> list[str]:
        """
        Generate responses for multiple prompts concurrently.
        
        Args:
            prompts: List of prompts to process
            max_concurrent: Maximum concurrent requests
            **kwargs: Generation parameters
            
        Returns:
            List of generated responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(prompt):
            async with semaphore:
                return await self.generate(prompt, **kwargs)
        
        tasks = [generate_with_semaphore(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_current_provider(self) -> str:
        """Get name of currently active LLM provider."""
        if self.providers and self.current_provider_index < len(self.providers):
            return self.providers[self.current_provider_index]["name"]
        return "none"
    
    def get_available_providers(self) -> List[str]:
        """Get list of all available LLM providers."""
        return [provider["name"] for provider in self.providers]
    
    def get_failed_providers(self) -> List[str]:
        """Get list of currently failed providers."""
        return list(self.failed_providers)
    
    def reset_failed_providers(self):
        """Reset failed providers list to retry them."""
        self.failed_providers.clear()
        logger.info("Reset failed providers - all providers available for retry")
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all providers."""
        return {
            "total_providers": len(self.providers),
            "available_providers": self.get_available_providers(),
            "current_provider": self.get_current_provider(),
            "failed_providers": self.get_failed_providers(),
            "provider_details": [
                {
                    "name": p["name"],
                    "model": p["model"],
                    "status": "failed" if p["name"] in self.failed_providers else "available"
                }
                for p in self.providers
            ]
        }