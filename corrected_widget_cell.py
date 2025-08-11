"""
This is the corrected widget cell code that should replace the existing widget function
in your notebook. The key addition is the bridge save functionality.
"""

# Interactive function (CORRECTED VERSION WITH BRIDGE)
def run_selected_fitment(b=None):
    with output_area:
        clear_output(wait=True)
        
        fitment_name = fitment_dropdown.value
        initial_kappa = kappa_slider.value
        target_param = target_param_dropdown.value
        max_iter = max_iter_slider.value
        
        print(f"🔧 Running {fitment_name.upper()} fitment...")
        print(f"   Initial κ: {initial_kappa}")
        
        # Get the fitment and run it
        fitter = get_fitment(fitment_name)
        initial_params = {"kappa": initial_kappa}
        
        if fitment_name == "no_fit":
            result = fitter.fit(demo_dataset, "ppn", {}, "exponential", initial_params)
        else:
            extra_params = {
                "target_param": target_param,
                "init": initial_kappa,
                "max_iter": max_iter
            }
            result = fitter.fit(
                demo_dataset, "ppn", {}, "exponential", initial_params,
                extra=extra_params
            )
        
        # Store result globally
        global current_fitment_result
        current_fitment_result = result
        
        # 🔗 BRIDGE INTEGRATION: Save state for experiments to use
        try:
            from tacc.core.experiment_bridge import save_fitment_state
            save_fitment_state(fitment_name, result, extra_params if fitment_name != "no_fit" else {})
            print("   🔗 State saved for experiments!")
        except ImportError:
            print("   ⚠️ Bridge not available")
        
        # Display results
        final_kappa = result['bn']['kappa']
        change = final_kappa - initial_kappa
        
        print(f"\n📊 RESULTS:")
        print(f"   Final κ: {final_kappa:.4f}")
        print(f"   Change: {change:+.4f}")
        
        # Test prediction
        test_input = MicrolawInput(phi=0.1)
        pred = compute_BN("ppn", test_input, {}, "exponential", result['bn'])
        
        print(f"\n🎯 PREDICTION (φ=0.1):")
        print(f"   N = {pred['N']:.3f}")
        print(f"   B = {pred['B']:.3f}")
        
        # Show fitment behavior
        if fitment_name == "no_fit":
            print("\n✓ No-fit: Parameters unchanged (as expected)")
        else:
            print(f"\n✓ {fitment_name}: Optimized κ after {max_iter} iterations")
