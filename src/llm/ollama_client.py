"""
Ollama Client - Local LLM inference wrapper for Gemma3:4B.

Features:
- Text generation with customizable parameters
- Vision support for image analysis
- Streaming support for long responses
- Retry logic for robustness
"""
import json
import time
from typing import Optional, Dict, Any, List, Generator
import logging
import base64                                                                     # "base64" is not accessed

try:
    import requests
except ImportError:
    requests = None

try:
    import httpx
except ImportError:
    httpx = None

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for Ollama local LLM inference.
    
    Designed for Gemma3:4B but compatible with other Ollama models.
    
    Features:
    - Synchronous generation
    - Vision/multimodal support
    - Configurable temperature and tokens
    - Retry logic for reliability
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma3:4b",
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Ollama API base URL
            model: Model name (e.g., "gemma3:4b")
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
        """
        if requests is None:
            raise ImportError("requests is required. Install with: pip install requests")
        
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Verify connection
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama is running and model is available."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()
            
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            # Check if our model is available (with or without tag)
            model_base = self.model.split(':')[0]
            available = any(
                model_base in name for name in model_names
            )
            
            if not available:
                logger.warning(
                    f"Model '{self.model}' may not be available. "
                    f"Available models: {model_names}"
                )
            else:
                logger.info(f"Ollama connected. Model: {self.model}")
                
        except requests.RequestException as e:
            logger.error(f"Could not connect to Ollama at {self.base_url}: {e}")
            raise ConnectionError(
                f"Ollama is not running or not accessible at {self.base_url}. "
                "Please start Ollama with: ollama serve"
            )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        images: Optional[List[str]] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            images: Optional list of base64-encoded images for vision
            stream: Whether to stream the response
            
        Returns:
            Generated text
        """
        # Build request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        if images:
            payload["images"] = images
        
        # Make request with retries
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if stream:
                    return self._generate_stream(payload)
                else:
                    return self._generate_sync(payload)
                    
            except requests.RequestException as e:
                last_error = e
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        raise RuntimeError(f"Failed after {self.max_retries} attempts: {last_error}")
    
    def _generate_sync(self, payload: Dict[str, Any]) -> str:
        """Synchronous generation."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '')
    
    def _generate_stream(self, payload: Dict[str, Any]) -> str:
        """Streaming generation (returns complete text)."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
            stream=True
        )
        response.raise_for_status()
        
        full_response = []
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if 'response' in data:
                        full_response.append(data['response'])
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
        
        return ''.join(full_response)
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> Generator[str, None, None]:
        """
        Generate text with streaming output.
        
        Yields tokens as they're generated.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if 'response' in data:
                        yield data['response']
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
    
    def analyze_image(
        self,
        image_base64: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3
    ) -> str:
        """
        Analyze an image with vision capabilities.
        
        Args:
            image_base64: Base64-encoded image
            prompt: Analysis prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            
        Returns:
            Image analysis text
        """
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            images=[image_base64]
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> str:
        """
        Chat completion with message history.
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Assistant response
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get('message', {}).get('content', '')
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"Could not get model info: {e}")
            return {"error": str(e)}
    
    def is_available(self) -> bool:
        """Check if Ollama and the model are available."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
