"""LM Studio API client for batch processing."""
import json
import time
import requests
from typing import Dict, Any, Optional, List
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


class LMStudioClient:
    """Client for interacting with LM Studio's OpenAI-compatible API."""
    
    def __init__(self, server_url: str, timeout: int = 30, retry_attempts: int = 3, retry_delay: float = 1.0):
        """Initialize LM Studio client.
        
        Args:
            server_url: LM Studio server URL
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
            retry_delay: Base delay between retries
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Set up session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=retry_attempts,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST"],
            backoff_factor=retry_delay
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # API endpoints
        self.chat_endpoint = f"{self.server_url}/v1/chat/completions"
        self.models_endpoint = f"{self.server_url}/v1/models"
    
    def health_check(self) -> bool:
        """Check if LM Studio server is running and accessible.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = self.session.get(self.models_endpoint, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from LM Studio.
        
        Returns:
            List of model information dictionaries
        
        Raises:
            Exception: If unable to fetch models
        """
        try:
            response = self.session.get(self.models_endpoint, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get('data', [])
        except Exception as e:
            raise Exception(f"Failed to fetch models: {str(e)}")
    
    def send_request(self, 
                    prompt: str, 
                    model: str = None,
                    temperature: float = 0.7,
                    max_tokens: int = 2048,
                    **kwargs) -> Dict[str, Any]:
        """Send completion request to LM Studio.
        
        Args:
            prompt: The prompt to send
            model: Model to use (if None, uses server default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters for the API
        
        Returns:
            Response dictionary containing the completion
        
        Raises:
            Exception: If request fails
        """
        # Prepare request payload
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            **kwargs
        }
        
        # Add model if specified
        if model and model != 'default':
            payload["model"] = model
        
        try:
            response = self.session.post(
                self.chat_endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            raise Exception("Request timed out. The model might be taking too long to respond.")
        
        except requests.exceptions.ConnectionError:
            raise Exception("Failed to connect to LM Studio. Make sure the server is running.")
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise Exception("API endpoint not found. Make sure LM Studio server is properly configured.")
            elif e.response.status_code == 400:
                try:
                    error_detail = e.response.json().get('error', e.response.text)
                    if 'context length' in error_detail.lower() or 'context overflow' in error_detail.lower():
                        raise Exception(f"Context length exceeded. Try using a shorter input or increase the model's context length: {error_detail}")
                    else:
                        raise Exception(f"Bad request: {error_detail}")
                except ValueError:
                    raise Exception(f"Bad request: {e.response.text}")
            elif e.response.status_code == 422:
                try:
                    error_detail = e.response.json().get('detail', 'Unknown validation error')
                    raise Exception(f"Request validation failed: {error_detail}")
                except:
                    raise Exception("Request validation failed. Check your parameters.")
            else:
                raise Exception(f"HTTP {e.response.status_code}: {e.response.text}")
        
        except json.JSONDecodeError:
            raise Exception("Invalid response from server. Make sure LM Studio is running correctly.")
        
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")
    
    def extract_response_text(self, response: Dict[str, Any]) -> str:
        """Extract the text content from an API response.
        
        Args:
            response: The API response dictionary
        
        Returns:
            The extracted text content
        
        Raises:
            Exception: If response format is unexpected
        """
        try:
            choices = response.get('choices', [])
            if not choices:
                raise Exception("No choices in response")
            
            message = choices[0].get('message', {})
            content = message.get('content', '')
            
            if not content:
                raise Exception("Empty response content")
            
            return content.strip()
        
        except Exception as e:
            raise Exception(f"Failed to extract response text: {str(e)}")
    
    def validate_connection(self) -> Dict[str, Any]:
        """Validate connection and return server info.
        
        Returns:
            Dictionary with connection status and server info
        """
        result = {
            'connected': False,
            'server_url': self.server_url,
            'models': [],
            'error': None
        }
        
        try:
            if not self.health_check():
                result['error'] = "Server is not responding"
                return result
            
            models = self.get_models()
            result['connected'] = True
            result['models'] = [model.get('id', 'unknown') for model in models]
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def close(self):
        """Close the session and cleanup resources."""
        if hasattr(self, 'session'):
            self.session.close()