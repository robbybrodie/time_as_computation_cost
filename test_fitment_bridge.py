"""
Test script to demonstrate fitment bridge connection between widgets and experiments.
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path().resolve()
sys.path.insert(0, str(repo_root / "src"))

from tacc.core.experiment_bridge import save_fitment_state, get_fitted_kappa, is_fitment_active, get_active_fitment_info
from tacc.core.nb import get_fitment

def demo_widget_to_experiment_connection():
    """Demonstrate how widget fitment results can control experiments."""
    
    print("🎮 FITMENT BRIDGE DEMONSTRATION")
    print("=" * 50)
    
    # Simulate widget interaction
    print("\n1️⃣ Simulating widget fitment run...")
    
    # Create demo dataset
    demo_dataset = [
        {"x": {"phi": 0.0}, "target": 1.0, "weight": 1.0},
        {"x": {"phi": 0.1}, "target": 1.05, "weight": 1.0},
        {"x": {"phi": 0.2}, "target": 1.10, "weight": 1.0}
    ]
    
    # Run single_param fitment
    fitter = get_fitment("single_param")
    result = fitter.fit(
        demo_dataset, "ppn", {}, "exponential", {"kappa": 2.0},
        extra={"target_param": "kappa", "init": 2.0, "max_iter": 20}
    )
    
    fitted_kappa = result['bn']['kappa']
    print(f"   Fitted κ: {fitted_kappa:.4f}")
    
    # Save state for experiments to use
    save_fitment_state("single_param", result, {"target_param": "kappa"})
    print("   ✅ State saved to bridge")
    
    print("\n2️⃣ Checking experiment access...")
    
    # Check if experiments can see the fitment
    if is_fitment_active():
        experiment_kappa = get_fitted_kappa()
        fitment_info = get_active_fitment_info()
        
        print(f"   Experiment sees fitment: {fitment_info['name']}")
        print(f"   Experiment uses κ: {experiment_kappa:.4f}")
        
        if abs(experiment_kappa - fitted_kappa) < 1e-6:
            print("   ✅ BRIDGE WORKING! Experiment gets same κ as widget")
        else:
            print("   ❌ Bridge broken - different κ values")
    else:
        print("   ❌ Experiment doesn't see active fitment")
    
    print("\n3️⃣ Testing experiment runner...")
    
    # Now run the actual experiment - it should use fitted kappa
    from experiments.run_ppn import main as run_ppn
    
    print("   Running PPN experiment (should use fitted κ)...")
    run_ppn()
    
    print("\n🎯 BRIDGE DEMO COMPLETE!")
    print("   Widgets can now control experiment parameters!")

if __name__ == "__main__":
    demo_widget_to_experiment_connection()
