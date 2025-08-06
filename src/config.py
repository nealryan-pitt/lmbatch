"""Configuration management for LM Batch processing."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for LM Batch."""
    
    DEFAULT_CONFIG = {
        'lm_studio': {
            'server_url': 'http://localhost:1234',
            'model': 'gpt-oss-20b',
            'timeout': 30,
            'retry_attempts': 3,
            'retry_delay': 1.0,
        },
        'processing': {
            'temperature': 0.1,
            'max_tokens': 32000,
            'concurrent_requests': 3,
            'chunk_size': 8192,
            'max_context_length': 16384,  # Larger default for therapeutic analysis
        },
        'context_handling': {
            'strategy': 'force',  # Send full content by default for therapeutic analysis
            'auto_detect': True,
            'safety_margin': 500,
            'overlap_tokens': 300,
            'warn_on_truncation': True,
            'ctx_size': 16384,  # Default LM Studio context size
        },
        'model_presets': {
            'gpt-oss-20b': 16384,
            'openai/gpt-oss-20b': 16384,
            'gpt-oss-120b': 16384,
            'openai/gpt-oss-120b': 16384,
            'llama-3.3-70b': 32768,
            'meta/llama-3.3-70b': 32768,
            'qwen2.5-72b-instruct': 32768,
            'qwen/qwen2.5-72b-instruct': 32768,
            'default': 16384,
        },
        'output': {
            'directory': 'output',
            'overwrite': False,
            'include_metadata': True,
        },
    }
    
    def __init__(self, config_path: str = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to config file. If None, looks for config.yaml
        """
        self.config_path = config_path or 'config.yaml'
        self._config = self.DEFAULT_CONFIG.copy()
        self._load_config()
        self._load_env_overrides()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f) or {}
                self._merge_config(self._config, user_config)
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_path}: {e}")
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables."""
        env_mappings = {
            'LMBATCH_SERVER_URL': ('lm_studio', 'server_url'),
            'LMBATCH_MODEL': ('lm_studio', 'model'),
            'LMBATCH_TIMEOUT': ('lm_studio', 'timeout'),
            'LMBATCH_TEMPERATURE': ('processing', 'temperature'),
            'LMBATCH_MAX_TOKENS': ('processing', 'max_tokens'),
            'LMBATCH_OUTPUT_DIR': ('output', 'directory'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Convert numeric values
                if key in ['timeout', 'max_tokens']:
                    value = int(value)
                elif key in ['temperature']:
                    value = float(value)
                elif key in ['overwrite', 'include_metadata']:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                
                self._config[section][key] = value
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, section: str, key: str = None, default=None):
        """Get configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key (if None, returns entire section)
            default: Default value if not found
        """
        if key is None:
            return self._config.get(section, default)
        return self._config.get(section, {}).get(key, default)
    
    def set(self, section: str, key: str, value: Any):
        """Set configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
        """
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
    
    def save(self, path: str = None):
        """Save configuration to file.
        
        Args:
            path: File path to save to (defaults to current config_path)
        """
        save_path = path or self.config_path
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise Exception(f"Could not save config to {save_path}: {e}")
    
    @property
    def lm_studio(self) -> Dict[str, Any]:
        """Get LM Studio configuration."""
        return self._config['lm_studio']
    
    @property
    def processing(self) -> Dict[str, Any]:
        """Get processing configuration."""
        return self._config['processing']
    
    @property
    def output(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self._config['output']
    
    @property
    def context_handling(self) -> Dict[str, Any]:
        """Get context handling configuration."""
        return self._config['context_handling']
    
    @property
    def model_presets(self) -> Dict[str, Any]:
        """Get model preset configurations."""
        return self._config['model_presets']
    
    def get_model_context_length(self, model_name: str) -> int:
        """Get context length for a specific model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Context length in tokens
        """
        presets = self.model_presets
        
        # Try exact match first
        if model_name in presets:
            return presets[model_name]
        
        # Try without provider prefix (e.g. "openai/gpt-oss-20b" -> "gpt-oss-20b")
        if '/' in model_name:
            base_name = model_name.split('/')[-1]
            if base_name in presets:
                return presets[base_name]
        
        # Try partial matches for common patterns
        for preset_name, context_length in presets.items():
            if preset_name != 'default' and preset_name in model_name:
                return context_length
        
        # Fall back to default
        return presets['default']