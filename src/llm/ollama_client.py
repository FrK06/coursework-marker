"""
Ollama Client - Local LLM inference wrapper.
"""
import json
import time
from typing import Optional, Dict, Any, List
import logging

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for Ollama local LLM inference."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma3:4b",
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        if requests is None:
            raise ImportError("requests is required. Install with: pip install requests")
        
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            model_base = self.model.split(':')[0]
            available = any(model_base in name for name in model_names)
            
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

    def extract_text_from_image(
        self,
        image_base64: str,
        prompt: str = "Extract all text, code, labels, and metrics from this image. Preserve formatting and structure.",
        model_override: Optional[str] = None
    ) -> str:
        """
        Extract text from an image using OCR model (e.g., glm-ocr).

        Args:
            image_base64: Base64-encoded image
            prompt: Instruction prompt for OCR
            model_override: Optional model to use (defaults to glm-ocr via config)

        Returns:
            Extracted text from image
        """
        from config import OCR_MODEL, OCR_ENABLED

        if not OCR_ENABLED:
            logger.debug("OCR is disabled in config")
            return ""

        ocr_model = model_override or OCR_MODEL

        payload = {
            "model": ocr_model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temp for accurate extraction
                "num_predict": 2000,  # Allow longer output for code/charts
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            extracted_text = result.get('response', '').strip()

            logger.debug(f"OCR extracted {len(extracted_text)} chars using {ocr_model}")
            return extracted_text

        except requests.RequestException as e:
            logger.warning(f"OCR extraction failed with {ocr_model}: {e}")
            return ""

    def is_available(self) -> bool:
        """Check if Ollama and the model are available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
