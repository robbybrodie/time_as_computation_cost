"""
Script to add the bridge functionality to the existing widget code.
This will make the widgets save their state for experiments to use.
"""

print("""
🔧 BRIDGE INTEGRATION INSTRUCTIONS

To make your fitment widgets control the actual experiments, add this line
to the widget function in your notebook after the fitment result is computed:

```python
# ADDED: Save state for experiments to use
try:
    from tacc.core.experiment_bridge import save_fitment_state
    save_fitment_state(fitment_name, result, extra_params)
    print("   🔗 State saved for experiments!")
except ImportError:
    print("   ⚠️ Bridge not available")
```

This should go right after:
```python
current_fitment_result = result
```

And before:
```python
# Display results
final_kappa = result['bn']['kappa']
```

This way, when you run a fitment in the widget, it automatically saves the
results for the experiments to pick up!

Then when you run an experiment like PPN, it will show:
🎯 USING FITMENT: single_param
   Fitted κ: 0.4786
   This will affect all calculations!

Instead of:
🔧 No active fitment - using default parameter range
""")

if __name__ == "__main__":
    print("\n🎮 Testing the bridge connection...")
    
    # Test the bridge
    import subprocess
    import sys
    
    try:
        result = subprocess.run([sys.executable, "test_fitment_bridge.py"], 
                              capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✅ Bridge test successful!")
            print("\nOutput:")
            print(result.stdout)
        else:
            print("❌ Bridge test failed:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Could not run bridge test: {e}")

    print("\n🎯 TO FIX THE WIDGET ISSUE:")
    print("Add the bridge save code shown above to your notebook widget function!")
