"""Main CLI entry point for TACC."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Import fitments to ensure they're registered
import tacc.fitments
from tacc.core.nb import compute_BN
from tacc.core.fitment import get as get_fitment
from tacc.core.microlaw import list_microlaws
from tacc.core.bn import list_bn_families


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = "configs/default.yaml"
        
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, ImportError):
        # Return default config if file not found or yaml not available
        return {
            'microlaw': 'ppn',
            'microlaw_params': {},
            'bn_family': 'exponential', 
            'bn_params': {'kappa': 2.0},
            'fitment': {'name': 'no_fit', 'params': {}}
        }


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
    with open(path, 'r') as f:
        data = json.load(f)
        
    # Handle both single object and list of objects
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Dataset must be a JSON object or array of objects")


def create_dummy_experiment() -> str:
    """Create a dummy experiment for testing."""
    return "dummy_experiment"


def run_ppn_experiment(ml_params: Dict[str, Any], bn_params: Dict[str, Any]) -> Dict[str, Any]:
    """Run a simple PPN experiment."""
    from tacc.core.microlaw import MicrolawInput
    
    # Create a simple test case
    inputs = MicrolawInput(phi=0.1)  # Weak gravitational field
    result = compute_BN("ppn", inputs, ml_params, "exponential", bn_params)
    
    return {
        'experiment': 'ppn',
        'inputs': inputs.to_dict(),
        'result': result
    }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TACC: Time As Computation Cost",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters
  python -m tacc.cli.main ppn
  
  # Use no_fit fitment (pass-through)
  python -m tacc.cli.main ppn --fitment no_fit
  
  # Use single_param fitment to fit kappa
  python -m tacc.cli.main ppn --fitment single_param --fitment-params '{"target_param":"kappa","init":2.0}' --dataset data/ppn_small.json
  
  # List available components
  python -m tacc.cli.main list
        """
    )
    
    parser.add_argument(
        "experiment", 
        nargs="?",
        default="ppn",
        help="Experiment to run (ppn, dummy, list)"
    )
    
    parser.add_argument(
        "--config",
        default=None,
        help="Configuration file path (default: configs/default.yaml)"
    )
    
    parser.add_argument(
        "--fitment", 
        default=None,
        help="Fitment to use (default: no_fit)"
    )
    
    parser.add_argument(
        "--fitment-params", 
        default="{}",
        help="Fitment parameters as JSON string"
    )
    
    parser.add_argument(
        "--dataset", 
        default=None,
        help="Dataset file path (JSON or JSONL)"
    )
    
    parser.add_argument(
        "--microlaw",
        default=None,
        help="Microlaw to use (overrides config)"
    )
    
    parser.add_argument(
        "--bn-family", 
        default=None,
        help="B(N) family to use (overrides config)"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["json", "yaml", "table"],
        default="json",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.experiment == "list":
        print("Available Components:")
        print("===================")
        print(f"Microlaws: {list(list_microlaws().keys())}")
        print(f"B(N) Families: {list(list_bn_families().keys())}")
        
        # Import fitments to ensure they're registered
        from tacc.core.fitment import list_all
        print(f"Fitments: {list(list_all().keys())}")
        return
    
    # Load configuration
    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        cfg = load_config()  # Use default
    
    # Extract parameters from config
    microlaw_name = args.microlaw or cfg.get("microlaw", "ppn")
    ml_params = cfg.get("microlaw_params", {})
    bn_family = args.bn_family or cfg.get("bn_family", "exponential")
    bn_params = cfg.get("bn_params", {"kappa": 2.0})
    
    # Determine fitment
    fitment_name = args.fitment or cfg.get("fitment", {}).get("name", "no_fit")
    
    try:
        fitment_params = json.loads(args.fitment_params) if args.fitment_params != "{}" else cfg.get("fitment", {}).get("params", {})
    except json.JSONDecodeError as e:
        print(f"Error parsing fitment parameters: {e}")
        sys.exit(1)
    
    # Load dataset if provided
    dataset = []
    if args.dataset:
        try:
            dataset = load_dataset(args.dataset)
            print(f"Loaded dataset with {len(dataset)} entries")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)
    
    # Apply fitment
    try:
        fitter = get_fitment(fitment_name)
        fitted = fitter.fit(
            dataset, 
            microlaw_name, 
            ml_params, 
            bn_family, 
            bn_params, 
            loss=None, 
            extra=fitment_params
        )
        ml_params = fitted.get("microlaw", ml_params)
        bn_params = fitted.get("bn", bn_params)
        
        # Print fitment results
        fitment_result = {
            "fitment": {
                "name": fitment_name,
                "params": fitment_params
            },
            "fitted_params": fitted,
            "dataset_size": len(dataset)
        }
        
        print("Fitment Results:")
        print("===============")
        if args.output_format == "json":
            print(json.dumps(fitment_result, indent=2))
        else:
            for key, value in fitment_result.items():
                print(f"{key}: {value}")
        print()
        
    except Exception as e:
        print(f"Error applying fitment '{fitment_name}': {e}")
        sys.exit(1)
    
    # Run experiment if specified
    experiment_result = None
    if args.experiment == "ppn":
        try:
            experiment_result = run_ppn_experiment(ml_params, bn_params)
        except Exception as e:
            print(f"Error running PPN experiment: {e}")
            sys.exit(1)
    elif args.experiment == "dummy":
        experiment_result = {
            "experiment": "dummy",
            "status": "completed",
            "microlaw": {"name": microlaw_name, "params": ml_params},
            "bn_family": {"name": bn_family, "params": bn_params}
        }
    
    # Print experiment results
    if experiment_result:
        print("Experiment Results:")
        print("==================")
        if args.output_format == "json":
            print(json.dumps(experiment_result, indent=2))
        else:
            for key, value in experiment_result.items():
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
