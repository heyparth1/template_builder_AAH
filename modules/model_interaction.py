# model_interaction.py - Gemini API Integration Module
# Adapted to use Google Gemini 3 Pro instead of OpenAI GPT

import logging
import time
import random
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    MAX_TOKENS_PER_MINUTE,
    MAX_REQUESTS_PER_MINUTE,
    RATE_LIMIT_BUFFER
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gemini model configuration
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class RateLimitError(Exception):
    """Custom exception for rate limit errors"""
    pass


@dataclass
class APIResponse:
    """Structured API response"""
    content: str
    model: str
    tokens_used: int
    finish_reason: str
    success: bool
    error: Optional[str] = None


class RateLimiter:
    """
    Rate limiter for API requests.
    """
    
    def __init__(self, max_requests_per_minute: int = 500, max_tokens_per_minute: int = 90000):
        self.max_requests = int(max_requests_per_minute * RATE_LIMIT_BUFFER)
        self.max_tokens = int(max_tokens_per_minute * RATE_LIMIT_BUFFER)
        self.request_timestamps: List[float] = []
        self.token_counts: List[tuple] = []
    
    def wait_if_needed(self, estimated_tokens: int = 1000) -> None:
        """Wait if we're approaching rate limits."""
        current_time = time.time()
        window_start = current_time - 60
        
        self.request_timestamps = [t for t in self.request_timestamps if t > window_start]
        self.token_counts = [(t, c) for t, c in self.token_counts if t > window_start]
        
        if len(self.request_timestamps) >= self.max_requests:
            wait_time = self.request_timestamps[0] - window_start + 1
            logger.info(f"Rate limit approaching: waiting {wait_time:.1f}s (requests)")
            time.sleep(max(0, wait_time))
        
        total_tokens = sum(c for _, c in self.token_counts) + estimated_tokens
        if total_tokens >= self.max_tokens:
            wait_time = self.token_counts[0][0] - window_start + 1
            logger.info(f"Rate limit approaching: waiting {wait_time:.1f}s (tokens)")
            time.sleep(max(0, wait_time))
    
    def record_request(self, tokens_used: int) -> None:
        """Record a completed request for rate limiting"""
        current_time = time.time()
        self.request_timestamps.append(current_time)
        self.token_counts.append((current_time, tokens_used))


class GPTClient:
    """
    Google Gemini API client with retry logic and rate limiting.
    Maintains the same interface as the original GPTClient for compatibility.
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env variable)
            model: Model to use (defaults to gemini-3-pro-preview)
        """
        self.api_key = api_key or GOOGLE_API_KEY
        self.model = model or GEMINI_MODEL
        
        # Initialize client - google-genai reads from GOOGLE_API_KEY env var automatically
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            # Will use GOOGLE_API_KEY from environment
            self.client = genai.Client()
        
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=MAX_REQUESTS_PER_MINUTE,
            max_tokens_per_minute=MAX_TOKENS_PER_MINUTE
        )
        
        logger.info(f"Initialized Gemini client with model: {self.model}")
    
    def _add_jitter(self, base_delay: float = 0.1) -> None:
        """Add random delay to prevent thundering herd"""
        jitter = random.uniform(0.1, 0.5)
        time.sleep(base_delay + jitter)
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((RateLimitError, ConnectionError))
    )
    def call_api(
        self,
        prompt: str,
        system_message: str = "You are a document analysis expert. Return only valid JSON responses.",
        max_tokens: int = 8192,
        temperature: float = 0.3,
        json_mode: bool = True
    ) -> APIResponse:
        """
        Call the Gemini API with retry logic.
        
        Args:
            prompt: User prompt
            system_message: System message for context
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            json_mode: Whether to enforce JSON response
            
        Returns:
            APIResponse object
        """
        # Estimate tokens and wait if needed
        estimated_tokens = len(prompt.split()) * 1.3 + max_tokens
        self.rate_limiter.wait_if_needed(int(estimated_tokens))
        
        # Add jitter
        self._add_jitter()
        
        try:
            # Combine system message with prompt
            full_prompt = f"{system_message}\n\n{prompt}"
            
            # Configure generation
            config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                thinking_config=types.ThinkingConfig(thinking_level="low")
            )
            
            # Add JSON response mime type if json_mode is enabled
            if json_mode:
                config.response_mime_type = "application/json"
            
            logger.debug(f"Calling Gemini API with {len(prompt)} chars")
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=config
            )
            
            content = response.text if response.text else ""
            
            # Estimate tokens (Gemini doesn't always return usage in same format)
            tokens_used = len(content.split()) + len(prompt.split())
            finish_reason = "stop"
            
            # Record for rate limiting
            self.rate_limiter.record_request(tokens_used)
            
            logger.debug(f"API response: ~{tokens_used} tokens")
            
            return APIResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                success=True
            )
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Gemini API error: {error_message}")
            
            # Check for rate limit error
            if "rate_limit" in error_message.lower() or "429" in error_message or "quota" in error_message.lower():
                raise RateLimitError(f"Rate limited: {error_message}")
            
            return APIResponse(
                content="",
                model=self.model,
                tokens_used=0,
                finish_reason="error",
                success=False,
                error=error_message
            )
    
    def classify_content(self, prompt: str) -> Dict[str, Any]:
        """
        Classify document content and return parsed JSON.
        
        Args:
            prompt: Classification prompt
            
        Returns:
            Parsed JSON response as dictionary
        """
        response = self.call_api(prompt)
        
        if not response.success:
            logger.error(f"Classification failed: {response.error}")
            return {"replacements": [], "error": response.error}
        
        try:
            # Parse JSON response
            content = response.content
            
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            content = content.strip()
            
            # Find JSON object using raw_decode to handle trailing text
            try:
                # Find start of object
                start_idx = content.find('{')
                if start_idx == -1:
                    raise json.JSONDecodeError("No JSON object found", content, 0)
                
                content_slice = content[start_idx:]
                result, end_idx = json.JSONDecoder().raw_decode(content_slice)
            except json.JSONDecodeError:
                # Fallback: manual extraction if raw_decode fails
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    json_match = content[start:end]
                    result = json.loads(json_match)
                else:
                    raise
            
            # Validate structure
            if "replacements" not in result:
                result = {"replacements": []}
            
            if not isinstance(result["replacements"], list):
                result["replacements"] = []
            
            logger.info(f"Classified {len(result['replacements'])} replacements")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response.content}")
            return {"replacements": [], "error": f"JSON parse error: {e}"}


def create_client(api_key: str = None, model: str = None) -> GPTClient:
    """
    Factory function to create Gemini client.
    
    Args:
        api_key: Optional API key override
        model: Optional model override
        
    Returns:
        Configured GPTClient (Gemini) instance
    """
    return GPTClient(api_key=api_key, model=model)
