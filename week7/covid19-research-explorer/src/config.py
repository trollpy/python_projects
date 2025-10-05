"""
Configuration management for COVID-19 Research Explorer
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'data': {
                'metadata_file': 'data/metadata.csv',
                'chunk_size': 10000,
                'sample_size': 50000,
                'cache_enabled': True
            },
            'analysis': {
                'top_n_journals': 15,
                'top_n_sources': 10,
                'top_n_words': 30,
                'word_min_length': 4,
                'stop_words': ['with', 'from', 'that', 'this', 'have', 'been']
            },
            'visualization': {
                'figure_width': 12,
                'figure_height': 6,
                'dpi': 300,
                'style': 'whitegrid'
            },
            'streamlit': {
                'page_title': 'COVID-19 Research Explorer',
                'page_icon': '🦠',
                'layout': 'wide'
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot notation path"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value by dot notation path"""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        
        config[keys[-1]] = value


# Global config instance
config = Config()