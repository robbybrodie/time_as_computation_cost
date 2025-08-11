"""
Unified configuration system for TACC.

Provides single source of truth for all settings across CLI and notebooks.
Supports profiles, environment overrides, and validation.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Union
from pathlib import Path
import yaml
import os
from dataclasses import dataclass, field


@dataclass
class TACCConfig:
    """TACC configuration container with validation."""
    
    # Core settings
    microlaw: str = "ppn"
    microlaw_params: Dict[str, Any] = field(default_factory=dict)
    
    bn_family: str = "exponential" 
    bn_params: Dict[str, Any] = field(default_factory=lambda: {"kappa": 2.0})
    
    fitment_name: str = "no_fit"
    fitment_params: Dict[str, Any] = field(default_factory=dict)
    
    # Experiment settings
    output_dir: str = "output"
    save_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 160
    
    # Advanced settings
    enable_validation: bool = True
    enable_stamping: bool = True
    seed_strategy: str = "parameter_dependent"  # or "fixed" or "random"
    
    # Notebook settings
    show_chooser_widgets: bool = True
    auto_display_components: bool = True
    widget_style: str = "compact"  # or "detailed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'microlaw': self.microlaw,
            'microlaw_params': self.microlaw_params,
            'bn_family': self.bn_family,
            'bn_params': self.bn_params,
            'fitment': {
                'name': self.fitment_name,
                'params': self.fitment_params
            },
            'experiment': {
                'output_dir': self.output_dir,
                'save_plots': self.save_plots,
                'plot_format': self.plot_format,
                'plot_dpi': self.plot_dpi
            },
            'advanced': {
                'enable_validation': self.enable_validation,
                'enable_stamping': self.enable_stamping,
                'seed_strategy': self.seed_strategy
            },
            'notebook': {
                'show_chooser_widgets': self.show_chooser_widgets,
                'auto_display_components': self.auto_display_components,
                'widget_style': self.widget_style
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TACCConfig':
        """Create config from dictionary."""
        config = cls()
        
        # Core settings
        config.microlaw = data.get('microlaw', config.microlaw)
        config.microlaw_params = data.get('microlaw_params', config.microlaw_params)
        config.bn_family = data.get('bn_family', config.bn_family)
        config.bn_params = data.get('bn_params', config.bn_params)
        
        # Fitment settings
        fitment = data.get('fitment', {})
        config.fitment_name = fitment.get('name', config.fitment_name)
        config.fitment_params = fitment.get('params', config.fitment_params)
        
        # Experiment settings
        experiment = data.get('experiment', {})
        config.output_dir = experiment.get('output_dir', config.output_dir)
        config.save_plots = experiment.get('save_plots', config.save_plots)
        config.plot_format = experiment.get('plot_format', config.plot_format)
        config.plot_dpi = experiment.get('plot_dpi', config.plot_dpi)
        
        # Advanced settings
        advanced = data.get('advanced', {})
        config.enable_validation = advanced.get('enable_validation', config.enable_validation)
        config.enable_stamping = advanced.get('enable_stamping', config.enable_stamping)
        config.seed_strategy = advanced.get('seed_strategy', config.seed_strategy)
        
        # Notebook settings
        notebook = data.get('notebook', {})
        config.show_chooser_widgets = notebook.get('show_chooser_widgets', config.show_chooser_widgets)
        config.auto_display_components = notebook.get('auto_display_components', config.auto_display_components)
        config.widget_style = notebook.get('widget_style', config.widget_style)
        
        return config


class ConfigManager:
    """Manages TACC configuration loading and environment overrides."""
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[TACCConfig] = None
    
    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_config_root(self) -> Path:
        """Get the config root directory."""
        # Try to find config directory relative to this file
        current_dir = Path(__file__).parent.parent.parent
        config_dir = current_dir / "configs"
        
        if config_dir.exists():
            return config_dir
            
        # Fallback to current working directory
        cwd_config = Path.cwd() / "configs"
        if cwd_config.exists():
            return cwd_config
            
        # Create default in current directory
        cwd_config.mkdir(exist_ok=True)
        return cwd_config
    
    def load_config(self, 
                   profile: str = "default",
                   config_path: Optional[Union[str, Path]] = None,
                   override_dict: Optional[Dict[str, Any]] = None) -> TACCConfig:
        """
        Load configuration with profile support and environment overrides.
        
        Args:
            profile: Configuration profile name (default: "default")
            config_path: Optional explicit path to config file
            override_dict: Optional dictionary to override specific values
            
        Returns:
            TACCConfig instance
        """
        if config_path is None:
            config_root = self.get_config_root()
            config_path = config_root / f"{profile}.yaml"
        else:
            config_path = Path(config_path)
        
        # Load base configuration
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
        else:
            # Create default config if it doesn't exist
            config_data = {}
            self._create_default_config(config_path)
        
        # Apply environment variable overrides
        config_data = self._apply_env_overrides(config_data)
        
        # Apply explicit overrides
        if override_dict:
            config_data = self._merge_dicts(config_data, override_dict)
        
        # Create and cache config
        self._config = TACCConfig.from_dict(config_data)
        return self._config
    
    def get_current_config(self) -> TACCConfig:
        """Get the current cached config, loading default if none exists."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def save_config(self, config: TACCConfig, profile: str = "default") -> None:
        """Save configuration to file."""
        config_root = self.get_config_root()
        config_path = config_root / f"{profile}.yaml"
        
        with open(config_path, 'w') as f:
            yaml.safe_dump(config.to_dict(), f, indent=2, sort_keys=True)
    
    def _create_default_config(self, config_path: Path) -> None:
        """Create default configuration file."""
        default_config = TACCConfig()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.safe_dump(default_config.to_dict(), f, indent=2, sort_keys=True)
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        # Map environment variables to config paths
        env_mappings = {
            'TACC_MICROLAW': ['microlaw'],
            'TACC_BN_FAMILY': ['bn_family'], 
            'TACC_KAPPA': ['bn_params', 'kappa'],
            'TACC_FITMENT': ['fitment', 'name'],
            'TACC_OUTPUT_DIR': ['experiment', 'output_dir'],
            'TACC_ENABLE_VALIDATION': ['advanced', 'enable_validation'],
            'TACC_ENABLE_STAMPING': ['advanced', 'enable_stamping'],
        }
        
        for env_var, path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if path[-1] in ['enable_validation', 'enable_stamping', 'save_plots']:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif path[-1] in ['kappa', 'plot_dpi']:
                    try:
                        value = float(value) if '.' in value else int(value)
                    except ValueError:
                        continue
                
                # Set nested value
                current = config_data
                for key in path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[path[-1]] = value
        
        return config_data
    
    def _merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result


# Global config manager instance
_config_manager = ConfigManager()

# Convenience functions
def load_config(profile: str = "default", **kwargs) -> TACCConfig:
    """Load configuration with specified profile."""
    return _config_manager.load_config(profile=profile, **kwargs)

def get_config() -> TACCConfig:
    """Get current configuration."""
    return _config_manager.get_current_config()

def save_config(config: TACCConfig, profile: str = "default") -> None:
    """Save configuration to file."""
    _config_manager.save_config(config, profile)

def list_profiles() -> list[str]:
    """List available configuration profiles."""
    config_root = _config_manager.get_config_root()
    profiles = []
    
    for yaml_file in config_root.glob("*.yaml"):
        profile_name = yaml_file.stem
        profiles.append(profile_name)
    
    return sorted(profiles)

def create_profile(profile_name: str, base_profile: str = "default") -> TACCConfig:
    """Create new configuration profile based on existing one."""
    base_config = load_config(profile=base_profile)
    save_config(base_config, profile=profile_name)
    return base_config
