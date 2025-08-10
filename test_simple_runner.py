"""
Simple test runner that doesn't require external dependencies.
This tests the basic structure and imports of our new experiments.
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test causal diamond
        from tacc.lattices.diamond import run_demo, run_experiment, create_diamond_lattice
        print("✓ Causal diamond module imports successfully")
    except ImportError as e:
        print(f"✗ Causal diamond import failed: {e}")
        return False
    
    try:
        # Test tension bandgaps
        from tacc.bandgaps.tension import run_demo, run_experiment, generate_synthetic_data
        print("✓ Tension bandgaps module imports successfully")
    except ImportError as e:
        print(f"✗ Tension bandgaps import failed: {e}")
        return False
    
    try:
        # Test mode crowding
        from tacc.crowding.mode_crowding import run_demo, run_experiment, generate_utilities
        print("✓ Mode crowding module imports successfully")
    except ImportError as e:
        print(f"✗ Mode crowding import failed: {e}")
        return False
    
    return True

def test_structure():
    """Test that file structure is correct."""
    print("\nTesting file structure...")
    
    required_files = [
        "src/tacc/lattices/__init__.py",
        "src/tacc/lattices/diamond.py",
        "src/tacc/bandgaps/__init__.py", 
        "src/tacc/bandgaps/tension.py",
        "src/tacc/crowding/__init__.py",
        "src/tacc/crowding/mode_crowding.py",
        "experiments/run_causal_diamond.py",
        "experiments/run_tension_bandgaps.py",
        "experiments/run_mode_crowding.py",
        "notebooks/00_Run_All_Experiments.ipynb",
        "notebooks/01_Causal_Diamond_Colab.ipynb", 
        "notebooks/02_Tension_Bandgaps_Colab.ipynb",
        "notebooks/03_Mode_Crowding_Colab.ipynb",
        "tests/test_causal_diamond.py",
        "tests/test_tension_bandgaps.py",
        "tests/test_mode_crowding.py",
        "pyproject.toml"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = repo_root / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_functions_exist():
    """Test that required functions exist in modules."""
    print("\nTesting function signatures...")
    
    # Only test if imports worked
    try:
        from tacc.lattices.diamond import run_demo, run_experiment
        assert callable(run_demo), "run_demo should be callable"
        assert callable(run_experiment), "run_experiment should be callable"
        print("✓ Causal diamond functions exist")
        
        from tacc.bandgaps.tension import run_demo, run_experiment  
        assert callable(run_demo), "run_demo should be callable"
        assert callable(run_experiment), "run_experiment should be callable"
        print("✓ Tension bandgaps functions exist")
        
        from tacc.crowding.mode_crowding import run_demo, run_experiment
        assert callable(run_demo), "run_demo should be callable" 
        assert callable(run_experiment), "run_experiment should be callable"
        print("✓ Mode crowding functions exist")
        
        return True
        
    except Exception as e:
        print(f"✗ Function test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("SIMPLE TEST RUNNER - TACC NEW EXPERIMENTS")
    print("=" * 60)
    
    # Note about dependencies
    print("\nNote: This test checks structure and imports without running experiments.")
    print("Full functionality requires: numpy, scipy, matplotlib, networkx, scikit-learn")
    print("In Colab, these will be installed automatically via the bootstrap cells.")
    print()
    
    tests = [
        ("File structure", test_structure),
        ("Module imports", test_imports), 
        ("Function signatures", test_functions_exist)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All structural tests PASSED!")
        print("The new experiment framework is properly structured.")
        print("Ready for use in Google Colab or with dependencies installed.")
    else:
        print(f"\n❌ {total - passed} tests FAILED!")
        print("Check the error messages above for details.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
