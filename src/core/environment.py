import os
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    from agents import set_default_openai_api, set_tracing_disabled
except ImportError:
    # Fallback for testing without agents package
    def set_default_openai_api(api_type): pass
    def set_tracing_disabled(disabled): pass

@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    default_model: str = "llama3.2"
    timeout: int = 30

class OllamaEnvironment:
    """Manages Ollama-OpenAI compatibility configuration"""

    @staticmethod
    def setup(config: Optional[OllamaConfig] = None) -> bool:
        """Configure OpenAI SDK for Ollama compatibility"""
        if config is None:
            config = OllamaConfig()

        try:
            os.environ["OPENAI_BASE_URL"] = config.base_url
            os.environ["OPENAI_API_KEY"] = config.api_key

            set_default_openai_api('chat_completions')
            set_tracing_disabled(True)

            return True
        except Exception as e:
            raise EnvironmentError(f"Failed to setup Ollama environment: {e}")

    @staticmethod
    def validate_connection(base_url: str = "http://localhost:11434") -> Dict[str, Any]:
        """Validate Ollama server connection"""
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            response.raise_for_status()

            models = response.json().get('models', [])
            return {
                'connected': True,
                'models': [m['name'] for m in models],
                'model_count': len(models),
                'server_status': 'healthy'
            }
        except requests.RequestException as e:
            return {
                'connected': False,
                'error': str(e),
                'server_status': 'unreachable'
            }