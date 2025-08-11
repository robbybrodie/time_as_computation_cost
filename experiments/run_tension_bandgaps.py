"""
Tension Bandgaps Experiment Runner - FIXED VERSION
No data leakage: frozen generator + proper train/val/test split + AICc/BIC + leakage guard
"""

import sys
from pathlib import Path

# Bootstrap path setup for Colab compatibility
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from tacc.experiments.tension_bandgaps import TBGenParams, run_tension_bandgaps


def main():
    """Run the tension bandgaps experiment with fixed methodology."""
    print("Running Tension Bandgaps Experiment (FIXED - No Leakage)...")
    print("=" * 60)
    
    # Set up output directory
    output_dir = Path(__file__).resolve().parent / "results"
    
    # Default parameters (frozen - NEVER affected by fitment)
    gen_params = TBGenParams(
        n_points=50,
        noise_sigma=0.05,
        a_true=2.0,
        beta_true=1.5,
        x_min=0.0,
        x_max=1.0,
        seed=42
    )
    
    # Check for fitment state from interactive widgets
    fitment_name = "no_fit"
    fitment_params = {}
    
    try:
        from tacc.core.experiment_bridge import get_fitted_kappa, is_fitment_active, get_active_fitment_info
        
        if is_fitment_active():
            fitment_info = get_active_fitment_info()
            fitted_kappa = get_fitted_kappa()
            
            print(f"🎯 FITMENT DETECTED: {fitment_info['name']}")
            print(f"   Fitted κ: {fitted_kappa:.4f}")
            print("   NOTE: Fitment affects MODEL SELECTION only, NOT data generation!")
            
            # Apply fitment to model selection only (example: polynomial degree)
            fitment_name = fitment_info['name']
            if fitment_name == "single_param":
                # Use fitted kappa to select polynomial degree (example of safe fitment)
                degree = max(2, min(5, int(fitted_kappa)))
                fitment_params = {"degree": degree}
                print(f"   Model fitment: polynomial degree = {degree} (based on κ)")
            
        else:
            print("🔧 No active fitment - using default model settings")
            
    except ImportError:
        print("🔧 Fitment bridge not available - using defaults")
    
    print(f"\n📊 FROZEN DATA GENERATION:")
    print(f"   Generator seed: {gen_params.seed}")
    print(f"   Sample size: {gen_params.n_points}")
    print(f"   Noise level: {gen_params.noise_sigma}")
    print(f"   True a: {gen_params.a_true}")
    print(f"   True β: {gen_params.beta_true}")
    print(f"   ⚠️  These parameters are NEVER modified by fitment!")
    
    # Run experiment with proper ML practices
    try:
        stamp = run_tension_bandgaps(
            gen_p=gen_params,
            families=("exponential", "polynomial", "power_law"),
            fitment=(fitment_name, fitment_params),
            result_dir=str(output_dir)
        )
        
        print(f"\n✅ EXPERIMENT COMPLETED!")
        print(f"   Results saved: {output_dir}/tension_bandgaps_seed{gen_params.seed}.json")
        print(f"   Git commit: {stamp['commit'][:8]}...")
        print(f"   Train/val/test split: {stamp['train_size']}/{stamp['val_size']}/{stamp['test_size']}")
        
        print(f"\n📈 MODEL COMPARISON (AICc - lower is better):")
        valid_families = [f for f in stamp['families'] if not f.get('invalid')]
        
        if valid_families:
            # Sort by AICc
            sorted_families = sorted(valid_families, key=lambda x: x['val']['AICc'])
            
            for i, family in enumerate(sorted_families):
                marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
                name = family['family'].capitalize()
                aicc = family['val']['AICc']
                aic = family['val']['AIC']
                bic = family['val']['BIC']
                test_mse = family['test']['mse']
                
                print(f"   {marker} {name:12} | AICc: {aicc:8.2f} | AIC: {aic:8.2f} | BIC: {bic:8.2f} | Test MSE: {test_mse:.4f}")
            
            best_model = stamp['best_by_AICc']
            print(f"\n🎯 BEST MODEL: {best_model.upper()} (by AICc)")
            
        else:
            print("   ❌ All models failed to fit!")
        
        print(f"\n✅ LEAKAGE PREVENTION:")
        print("   ✓ Data generator frozen (independent of fitment)")
        print("   ✓ Train/val/test split enforced")  
        print("   ✓ AICc used for model selection")
        print("   ✓ Power-law fit in log-log space")
        print("   ✓ Results stamped with git commit + metadata")
        print("   ✓ Leakage guard passed!")
        
        return stamp
        
    except RuntimeError as e:
        if "LEAKAGE" in str(e):
            print(f"\n❌ LEAKAGE DETECTED: {e}")
            print("   This indicates the experiment has data leakage issues!")
            raise
        else:
            print(f"\n❌ EXPERIMENT FAILED: {e}")
            raise


if __name__ == "__main__":
    main()
