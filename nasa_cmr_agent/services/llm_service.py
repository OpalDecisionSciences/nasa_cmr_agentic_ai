from typing import Optional, Dict, Any, Union
import asyncio
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import settings


class LLMService:
    """
    Multi-provider LLM service with fallback mechanisms.
    
    Supports OpenAI and Anthropic with automatic fallback,
    rate limiting, and error handling.
    """
    
    def __init__(self):
        self.primary_llm = self._initialize_primary_llm()
        self.fallback_llm = self._initialize_fallback_llm()
        self.current_provider = "primary"
    
    def _initialize_primary_llm(self) -> Optional[Union[ChatOpenAI, ChatAnthropic]]:
        """Initialize primary LLM (OpenAI by default)."""
        try:
            if settings.openai_api_key:
                return ChatOpenAI(
                    api_key=settings.openai_api_key,
                    model=settings.openai_model,
                    temperature=0.1,
                    max_tokens=1000,
                    timeout=30
                )
        except Exception as e:
            print(f"Failed to initialize OpenAI LLM: {e}")
        
        return None
    
    def _initialize_fallback_llm(self) -> Optional[Union[ChatOpenAI, ChatAnthropic]]:
        """Initialize fallback LLM (Anthropic)."""
        try:
            if settings.anthropic_api_key:
                return ChatAnthropic(
                    api_key=settings.anthropic_api_key,
                    model=settings.anthropic_model,
                    temperature=0.1,
                    max_tokens=1000,
                    timeout=30
                )
        except Exception as e:
            print(f"Failed to initialize Anthropic LLM: {e}")
        
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
        Generate response using LLM with fallback mechanism.
        
        Args:
            prompt: Input prompt or message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
            
        Raises:
            Exception: If both primary and fallback LLMs fail
        """
        # Try primary LLM first
        if self.primary_llm and self.current_provider != "fallback":
            try:
                return await self._generate_with_llm(
                    self.primary_llm, prompt, max_tokens, temperature, **kwargs
                )
            except Exception as e:
                print(f"Primary LLM failed: {e}")
                if self.fallback_llm:
                    print("Switching to fallback LLM")
                    self.current_provider = "fallback"
                else:
                    raise
        
        # Try fallback LLM
        if self.fallback_llm:
            try:
                return await self._generate_with_llm(
                    self.fallback_llm, prompt, max_tokens, temperature, **kwargs
                )
            except Exception as e:
                print(f"Fallback LLM failed: {e}")
                # Reset to primary for next attempt
                self.current_provider = "primary"
                raise
        
        raise Exception("No available LLM providers")
    
    async def _generate_with_llm(
        self,
        llm: Union[ChatOpenAI, ChatAnthropic],
        prompt: Union[str, BaseMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate response with specific LLM."""
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
        if self.current_provider == "primary" and self.primary_llm:
            if isinstance(self.primary_llm, ChatOpenAI):
                return "openai"
            elif isinstance(self.primary_llm, ChatAnthropic):
                return "anthropic"
        elif self.current_provider == "fallback" and self.fallback_llm:
            if isinstance(self.fallback_llm, ChatOpenAI):
                return "openai"
            elif isinstance(self.fallback_llm, ChatAnthropic):
                return "anthropic"
        
        return "none"
    
    def reset_provider(self):
        """Reset to primary provider."""
        self.current_provider = "primary"