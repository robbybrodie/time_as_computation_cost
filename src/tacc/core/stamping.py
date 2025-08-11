"""
Result stamping system for TACC.

Automatically captures metadata for all experiment outputs to ensure reproducibility:
- Git commit hash and repository state
- Microlaw and B(N) family configurations
- Fitment parameters and settings
- Random seeds and dataset information
- Execution environment details
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import os
import sys
import platform
import hashlib

from .config import get_config


class MetadataStamper:
    """Captures and manages experimental metadata."""
    
    def __init__(self):
        self._git_info_cache = None
        
    def get_git_info(self) -> Dict[str, Any]:
        """Get git repository information."""
        if self._git_info_cache is not None:
            return self._git_info_cache
            
        git_info = {
            "commit_hash": None,
            "branch": None,
            "is_dirty": None,
            "remote_url": None,
            "commit_message": None,
            "commit_date": None,
            "tag": None
        }
        
        try:
            # Get current commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                git_info["commit_hash"] = result.stdout.strip()
                
            # Get current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                git_info["branch"] = result.stdout.strip()
                
            # Check if working directory is dirty
            result = subprocess.run(
                ["git", "diff-index", "--quiet", "HEAD", "--"],
                capture_output=True, timeout=5
            )
            git_info["is_dirty"] = result.returncode != 0
            
            # Get remote URL
            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                git_info["remote_url"] = result.stdout.strip()
                
            # Get commit message
            if git_info["commit_hash"]:
                result = subprocess.run(
                    ["git", "log", "-1", "--pretty=%s", git_info["commit_hash"]],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    git_info["commit_message"] = result.stdout.strip()
                    
                # Get commit date
                result = subprocess.run(
                    ["git", "log", "-1", "--pretty=%cd", "--date=iso", git_info["commit_hash"]],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    git_info["commit_date"] = result.stdout.strip()
                    
            # Get latest tag
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                git_info["tag"] = result.stdout.strip()
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Git not available or timeout
            pass
            
        self._git_info_cache = git_info
        return git_info
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get execution environment information."""
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "hostname": platform.node(),
            "user": os.getenv("USER", "unknown"),
            "working_directory": str(Path.cwd()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "timezone": str(datetime.now().astimezone().tzinfo),
            "environment_variables": {
                key: os.getenv(key) for key in [
                    "TACC_MICROLAW", "TACC_BN_FAMILY", "TACC_KAPPA", 
                    "TACC_FITMENT", "TACC_OUTPUT_DIR", "TACC_PROFILE"
                ] if os.getenv(key) is not None
            }
        }
    
    def get_package_info(self) -> Dict[str, Any]:
        """Get Python package version information."""
        package_info = {
            "tacc_version": "dev",  # Would be set from package metadata in real deployment
            "dependencies": {}
        }
        
        # Try to get key package versions
        try:
            import numpy
            package_info["dependencies"]["numpy"] = numpy.__version__
        except ImportError:
            pass
            
        try:
            import matplotlib
            package_info["dependencies"]["matplotlib"] = matplotlib.__version__
        except ImportError:
            pass
            
        try:
            import scipy
            package_info["dependencies"]["scipy"] = scipy.__version__
        except ImportError:
            pass
            
        try:
            import yaml
            package_info["dependencies"]["pyyaml"] = yaml.__version__
        except ImportError:
            pass
            
        return package_info
    
    def create_experiment_stamp(self,
                              experiment_name: str,
                              microlaw_name: str,
                              microlaw_params: Dict[str, Any],
                              bn_family_name: str, 
                              bn_params: Dict[str, Any],
                              fitment_name: str,
                              fitment_params: Dict[str, Any],
                              experiment_params: Optional[Dict[str, Any]] = None,
                              custom_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create comprehensive experiment metadata stamp."""
        
        config = get_config()
        
        stamp = {
            "stamp_version": "1.0",
            "experiment": {
                "name": experiment_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "parameters": experiment_params or {}
            },
            "physics_configuration": {
                "microlaw": {
                    "name": microlaw_name,
                    "parameters": microlaw_params
                },
                "bn_family": {
                    "name": bn_family_name,
                    "parameters": bn_params
                },
                "fitment": {
                    "name": fitment_name,
                    "parameters": fitment_params
                }
            },
            "system_configuration": {
                "config_profile": getattr(config, '_profile', 'default'),
                "seed_strategy": config.seed_strategy,
                "validation_enabled": config.enable_validation,
                "stamping_enabled": config.enable_stamping
            },
            "reproducibility": {
                "git": self.get_git_info(),
                "environment": self.get_environment_info(),
                "packages": self.get_package_info()
            }
        }
        
        # Add custom metadata if provided
        if custom_metadata:
            stamp["custom"] = custom_metadata
            
        # Generate stamp hash for integrity
        stamp_str = json.dumps(stamp, sort_keys=True, default=str)
        stamp["stamp_hash"] = hashlib.sha256(stamp_str.encode()).hexdigest()[:16]
        
        return stamp
    
    def save_stamp(self, stamp: Dict[str, Any], output_path: Path) -> Path:
        """Save experiment stamp to file."""
        stamp_path = output_path / "experiment_metadata.json"
        stamp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(stamp_path, 'w') as f:
            json.dump(stamp, f, indent=2, default=str, sort_keys=True)
            
        return stamp_path
    
    def load_stamp(self, stamp_path: Path) -> Dict[str, Any]:
        """Load experiment stamp from file."""
        with open(stamp_path, 'r') as f:
            return json.load(f)
    
    def verify_stamp_integrity(self, stamp: Dict[str, Any]) -> bool:
        """Verify that stamp hasn't been tampered with."""
        if "stamp_hash" not in stamp:
            return False
            
        expected_hash = stamp.pop("stamp_hash")
        stamp_str = json.dumps(stamp, sort_keys=True, default=str)
        computed_hash = hashlib.sha256(stamp_str.encode()).hexdigest()[:16]
        
        # Restore hash to stamp
        stamp["stamp_hash"] = expected_hash
        
        return expected_hash == computed_hash
    
    def compare_stamps(self, stamp1: Dict[str, Any], stamp2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two experiment stamps and report differences."""
        differences = {
            "identical": True,
            "physics_differences": [],
            "system_differences": [],
            "git_differences": []
        }
        
        # Compare physics configuration
        physics1 = stamp1.get("physics_configuration", {})
        physics2 = stamp2.get("physics_configuration", {})
        
        for component in ["microlaw", "bn_family", "fitment"]:
            if physics1.get(component) != physics2.get(component):
                differences["physics_differences"].append({
                    "component": component,
                    "stamp1": physics1.get(component),
                    "stamp2": physics2.get(component)
                })
                differences["identical"] = False
        
        # Compare git information
        git1 = stamp1.get("reproducibility", {}).get("git", {})
        git2 = stamp2.get("reproducibility", {}).get("git", {})
        
        for field in ["commit_hash", "branch", "is_dirty"]:
            if git1.get(field) != git2.get(field):
                differences["git_differences"].append({
                    "field": field,
                    "stamp1": git1.get(field),
                    "stamp2": git2.get(field)
                })
                differences["identical"] = False
        
        return differences


class StampManager:
    """Manages stamping for experiment runs."""
    
    def __init__(self):
        self.stamper = MetadataStamper()
        
    def stamp_experiment(self,
                        experiment_name: str,
                        output_dir: Path,
                        microlaw_name: str,
                        microlaw_params: Dict[str, Any],
                        bn_family_name: str,
                        bn_params: Dict[str, Any],
                        fitment_name: str = "no_fit",
                        fitment_params: Optional[Dict[str, Any]] = None,
                        experiment_params: Optional[Dict[str, Any]] = None,
                        custom_metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Create and save experiment stamp."""
        
        if fitment_params is None:
            fitment_params = {}
            
        stamp = self.stamper.create_experiment_stamp(
            experiment_name=experiment_name,
            microlaw_name=microlaw_name,
            microlaw_params=microlaw_params,
            bn_family_name=bn_family_name,
            bn_params=bn_params,
            fitment_name=fitment_name,
            fitment_params=fitment_params,
            experiment_params=experiment_params,
            custom_metadata=custom_metadata
        )
        
        return self.stamper.save_stamp(stamp, output_dir)
    
    def create_replay_script(self, stamp_path: Path) -> Path:
        """Create a script that can replay the exact experiment."""
        stamp = self.stamper.load_stamp(stamp_path)
        
        script_lines = [
            "#!/usr/bin/env python3",
            '"""',
            'Experiment replay script generated from metadata stamp.',
            f'Original experiment: {stamp["experiment"]["name"]}',
            f'Original timestamp: {stamp["experiment"]["timestamp"]}',
            f'Commit: {stamp["reproducibility"]["git"]["commit_hash"]}',
            '"""',
            "",
            "from tacc.core.nb import compute_BN",
            "from tacc.core.microlaw import MicrolawInput",
            "",
            "# Reproduce exact configuration",
        ]
        
        physics = stamp["physics_configuration"]
        
        # Add physics configuration
        script_lines.extend([
            f"microlaw_name = '{physics['microlaw']['name']}'",
            f"microlaw_params = {repr(physics['microlaw']['parameters'])}",
            f"bn_family_name = '{physics['bn_family']['name']}'", 
            f"bn_params = {repr(physics['bn_family']['parameters'])}",
            f"fitment_name = '{physics['fitment']['name']}'",
            f"fitment_params = {repr(physics['fitment']['parameters'])}",
            "",
            "# Example computation (modify inputs as needed)",
            "inputs = MicrolawInput(phi=0.1)",
            "result = compute_BN(",
            "    microlaw_name, inputs, microlaw_params,",
            "    bn_family_name, bn_params",
            ")",
            "",
            "print(f'N = {result[\"N\"]:.6f}')",
            "print(f'B = {result[\"B\"]:.6f}')",
        ])
        
        script_path = stamp_path.parent / "replay_experiment.py"
        script_content = "\n".join(script_lines)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        # Make executable
        script_path.chmod(0o755)
        
        return script_path


# Global stamp manager
_stamp_manager = StampManager()

# Convenience functions
def stamp_experiment(experiment_name: str, output_dir: Path, **kwargs) -> Path:
    """Create and save experiment stamp."""
    return _stamp_manager.stamp_experiment(experiment_name, output_dir, **kwargs)

def create_replay_script(stamp_path: Path) -> Path:
    """Create replay script from stamp."""
    return _stamp_manager.create_replay_script(stamp_path)

def verify_experiment_stamp(stamp_path: Path) -> bool:
    """Verify experiment stamp integrity."""
    stamper = MetadataStamper()
    stamp = stamper.load_stamp(stamp_path)
    return stamper.verify_stamp_integrity(stamp)

def compare_experiment_stamps(stamp_path1: Path, stamp_path2: Path) -> Dict[str, Any]:
    """Compare two experiment stamps."""
    stamper = MetadataStamper()
    stamp1 = stamper.load_stamp(stamp_path1)
    stamp2 = stamper.load_stamp(stamp_path2)
    return stamper.compare_stamps(stamp1, stamp2)
