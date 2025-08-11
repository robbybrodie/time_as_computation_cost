"""
Interactive notebook widgets for TACC component selection.

Provides visual chooser widgets for microlaws, B(N) families, and fitments
to enable one-click parameter switching in Jupyter notebooks.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import json

from .microlaw import list_microlaws
from .bn import list_bn_families, get_bn_family_info
from .fitment import list_all as list_fitments
from .config import get_config, load_config, list_profiles


def create_component_chooser() -> str:
    """Create HTML/JavaScript component chooser widget."""
    
    # Get available components
    microlaws = list_microlaws()
    bn_families = get_bn_family_info()
    fitments = list_fitments()
    profiles = list_profiles()
    
    config = get_config()
    
    html = f"""
<div id="tacc-chooser" style="
    border: 2px solid #007acc; 
    border-radius: 10px; 
    padding: 20px; 
    margin: 10px 0; 
    background: #f8f9fa;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
">
    <h3 style="margin-top: 0; color: #007acc;">🧮 TACC Component Chooser</h3>
    
    <!-- Configuration Profile -->
    <div style="margin-bottom: 15px;">
        <label style="font-weight: bold; display: block; margin-bottom: 5px;">
            📋 Configuration Profile:
        </label>
        <select id="tacc-profile" style="
            padding: 8px; border: 1px solid #ddd; border-radius: 5px; 
            background: white; min-width: 200px;
        ">
            {_create_profile_options(profiles, 'default')}
        </select>
    </div>
    
    <!-- Microlaw Selection -->
    <div style="margin-bottom: 15px;">
        <label style="font-weight: bold; display: block; margin-bottom: 5px;">
            ⚛️ Microlaw (Physical Invariants → N):
        </label>
        <select id="tacc-microlaw" style="
            padding: 8px; border: 1px solid #ddd; border-radius: 5px; 
            background: white; min-width: 200px;
        ">
            {_create_microlaw_options(microlaws, config.microlaw)}
        </select>
        <div id="microlaw-info" style="
            margin-top: 5px; font-size: 0.9em; color: #666; 
            font-style: italic;
        "></div>
    </div>
    
    <!-- B(N) Family Selection -->
    <div style="margin-bottom: 15px;">
        <label style="font-weight: bold; display: block; margin-bottom: 5px;">
            📈 B(N) Family (N → Geometry Factor):
        </label>
        <select id="tacc-bn-family" style="
            padding: 8px; border: 1px solid #ddd; border-radius: 5px; 
            background: white; min-width: 200px;
        ">
            {_create_bn_family_options(bn_families, config.bn_family)}
        </select>
        <div id="bn-family-info" style="
            margin-top: 5px; font-size: 0.9em; color: #666; 
            font-style: italic;
        "></div>
        <div id="bn-params" style="margin-top: 10px;">
            {_create_bn_params_controls(config.bn_family, config.bn_params)}
        </div>
    </div>
    
    <!-- Fitment Selection -->
    <div style="margin-bottom: 15px;">
        <label style="font-weight: bold; display: block; margin-bottom: 5px;">
            🎯 Fitment (Parameter Optimization):
        </label>
        <select id="tacc-fitment" style="
            padding: 8px; border: 1px solid #ddd; border-radius: 5px; 
            background: white; min-width: 200px;
        ">
            {_create_fitment_options(fitments, config.fitment_name)}
        </select>
        <div id="fitment-info" style="
            margin-top: 5px; font-size: 0.9em; color: #666; 
            font-style: italic;
        "></div>
    </div>
    
    <!-- Action Buttons -->
    <div style="margin-top: 20px; text-align: center;">
        <button id="tacc-apply" style="
            background: #007acc; color: white; border: none; 
            padding: 10px 20px; border-radius: 5px; 
            font-size: 1em; cursor: pointer; margin-right: 10px;
        ">
            ✅ Apply Configuration
        </button>
        <button id="tacc-test" style="
            background: #28a745; color: white; border: none; 
            padding: 10px 20px; border-radius: 5px; 
            font-size: 1em; cursor: pointer; margin-right: 10px;
        ">
            🧪 Test Computation
        </button>
        <button id="tacc-reset" style="
            background: #6c757d; color: white; border: none; 
            padding: 10px 20px; border-radius: 5px; 
            font-size: 1em; cursor: pointer;
        ">
            🔄 Reset to Default
        </button>
    </div>
    
    <!-- Status Display -->
    <div id="tacc-status" style="
        margin-top: 15px; padding: 10px; background: #e9ecef; 
        border-radius: 5px; font-family: monospace; font-size: 0.9em;
        display: none;
    "></div>
</div>

<script>
// Component information
const componentInfo = {{
    microlaws: {json.dumps({name: {"doc": ml.__doc__ or "No description"} for name, ml in microlaws.items()})},
    bn_families: {json.dumps(bn_families)},
    fitments: {json.dumps({name: {"doc": "Fitment: " + name} for name in fitments})}
}};

// Current configuration
let currentConfig = {{
    microlaw: '{config.microlaw}',
    bn_family: '{config.bn_family}',
    bn_params: {json.dumps(config.bn_params)},
    fitment: '{config.fitment_name}'
}};

// Event handlers
document.getElementById('tacc-microlaw').addEventListener('change', function() {{
    const selected = this.value;
    const info = componentInfo.microlaws[selected];
    document.getElementById('microlaw-info').textContent = info ? info.doc : '';
    currentConfig.microlaw = selected;
}});

document.getElementById('tacc-bn-family').addEventListener('change', function() {{
    const selected = this.value;
    const info = componentInfo.bn_families[selected];
    document.getElementById('bn-family-info').textContent = info ? info.docstring : '';
    currentConfig.bn_family = selected;
    updateBNParams(selected);
}});

document.getElementById('tacc-fitment').addEventListener('change', function() {{
    const selected = this.value;
    const info = componentInfo.fitments[selected];
    document.getElementById('fitment-info').textContent = info ? info.doc : '';
    currentConfig.fitment = selected;
}});

function updateBNParams(familyName) {{
    const paramsDiv = document.getElementById('bn-params');
    
    // Default parameters for each family
    const defaultParams = {{
        'exponential': {{'kappa': 2.0}},
        'power': {{'alpha': 1.0}},
        'linear': {{'a': 1.0, 'b': 0.0}},
        'sigmoid': {{'A': 1.0, 'k': 10.0, 'N0': 0.5, 'B0': 0.0}},
        'piecewise': {{'breakpoints': [[0.0, 0.0], [1.0, 1.0]]}},
        'polynomial': {{'coefficients': [0.0, 1.0]}},
        'logarithmic': {{'a': 1.0, 'b': 1.0, 'c': 0.0, 'd': 0.0}},
        'custom': {{'function': 'N', 'func_params': {{}}}}
    }};
    
    const params = defaultParams[familyName] || {{}};
    currentConfig.bn_params = params;
    
    let html = '<div style="font-size: 0.9em; color: #666;">Parameters: ';
    for (const [key, value] of Object.entries(params)) {{
        if (typeof value === 'number') {{
            html += `${{key}}=${{value}} `;
        }} else if (Array.isArray(value)) {{
            html += `${{key}}=[...] `;
        }} else {{
            html += `${{key}}=... `;
        }}
    }}
    html += '</div>';
    paramsDiv.innerHTML = html;
}}

document.getElementById('tacc-apply').addEventListener('click', function() {{
    const status = document.getElementById('tacc-status');
    status.style.display = 'block';
    status.style.background = '#d4edda';
    status.style.color = '#155724';
    status.innerHTML = `
        ✅ Configuration Applied:<br>
        Microlaw: ${{currentConfig.microlaw}}<br>
        B(N) Family: ${{currentConfig.bn_family}}<br>
        Parameters: ${{JSON.stringify(currentConfig.bn_params)}}<br>
        Fitment: ${{currentConfig.fitment}}<br><br>
        Ready to use: <code>tacc.core.nb.compute_BN(...)</code>
    `;
}});

document.getElementById('tacc-test').addEventListener('click', function() {{
    const status = document.getElementById('tacc-status');
    status.style.display = 'block';
    status.style.background = '#cce5ff';
    status.style.color = '#004085';
    status.innerHTML = `
        🧪 Test Code:<br>
        <code>
        from tacc.core.nb import compute_BN<br>
        from tacc.core.microlaw import MicrolawInput<br><br>
        inputs = MicrolawInput(phi=0.1)<br>
        result = compute_BN('<br>
        &nbsp;&nbsp;&nbsp;&nbsp;"${{currentConfig.microlaw}}", inputs, {{}},<br>
        &nbsp;&nbsp;&nbsp;&nbsp;"${{currentConfig.bn_family}}", ${{JSON.stringify(currentConfig.bn_params)}}<br>
        )<br>
        print(f"N = {{result['N']:.4f}}, B = {{result['B']:.4f}}")
        </code>
    `;
}});

document.getElementById('tacc-reset').addEventListener('click', function() {{
    document.getElementById('tacc-microlaw').value = '{config.microlaw}';
    document.getElementById('tacc-bn-family').value = '{config.bn_family}';
    document.getElementById('tacc-fitment').value = '{config.fitment_name}';
    
    currentConfig = {{
        microlaw: '{config.microlaw}',
        bn_family: '{config.bn_family}',
        bn_params: {json.dumps(config.bn_params)},
        fitment: '{config.fitment_name}'
    }};
    
    updateBNParams('{config.bn_family}');
    
    const status = document.getElementById('tacc-status');
    status.style.display = 'block';
    status.style.background = '#f8d7da';
    status.style.color = '#721c24';
    status.innerHTML = '🔄 Reset to default configuration';
    
    setTimeout(() => {{
        status.style.display = 'none';
    }}, 2000);
}});

// Initialize
updateBNParams('{config.bn_family}');
</script>
"""
    
    return html


def _create_profile_options(profiles: list, current: str) -> str:
    """Create HTML option elements for profiles."""
    options = []
    for profile in profiles:
        selected = "selected" if profile == current else ""
        options.append(f'<option value="{profile}" {selected}>{profile.title()}</option>')
    return "\n".join(options)


def _create_microlaw_options(microlaws: Dict[str, Any], current: str) -> str:
    """Create HTML option elements for microlaws."""
    options = []
    for name, ml in microlaws.items():
        selected = "selected" if name == current else ""
        doc = (ml.__doc__ or "").split('\n')[0][:50]
        options.append(f'<option value="{name}" {selected}>{name.upper()} - {doc}...</option>')
    return "\n".join(options)


def _create_bn_family_options(bn_families: Dict[str, Dict[str, Any]], current: str) -> str:
    """Create HTML option elements for B(N) families."""
    options = []
    for name, info in bn_families.items():
        selected = "selected" if name == current else ""
        doc = info.get("docstring", "").split('\n')[0][:50]
        options.append(f'<option value="{name}" {selected}>{name.title()} - {doc}...</option>')
    return "\n".join(options)


def _create_fitment_options(fitments: Dict[str, Any], current: str) -> str:
    """Create HTML option elements for fitments."""
    options = []
    for name in fitments:
        selected = "selected" if name == current else ""
        display_name = name.replace('_', ' ').title()
        options.append(f'<option value="{name}" {selected}>{display_name}</option>')
    return "\n".join(options)


def _create_bn_params_controls(family_name: str, params: Dict[str, Any]) -> str:
    """Create parameter control HTML for B(N) family."""
    param_strs = []
    for key, value in params.items():
        if isinstance(value, (int, float)):
            param_strs.append(f"{key}={value}")
        else:
            param_strs.append(f"{key}=...")
    
    if param_strs:
        return f'<div style="font-size: 0.9em; color: #666;">Parameters: {" ".join(param_strs)}</div>'
    return '<div style="font-size: 0.9em; color: #666;">No parameters</div>'


def create_sanity_check_cell() -> str:
    """Create a sanity check cell that validates the complete pipeline."""
    
    return '''
# TACC Sanity Check Cell
# Run this cell to verify that all components are working correctly

from tacc.core.nb import compute_BN, list_components
from tacc.core.microlaw import MicrolawInput
from tacc.core.config import get_config
from tacc.core.validation import validate_complete_setup
from tacc.core.stamping import stamp_experiment
from pathlib import Path

print("🧮 TACC System Sanity Check")
print("=" * 40)

# 1. Check component availability
print("\\n1️⃣ Component Availability:")
components = list_components()
print(f"   Microlaws: {len(components['microlaws'])} available")
print(f"   B(N) Families: {len(components['bn_families'])} available") 
print(f"   Fitments: {len(components['fitments'])} available")

# 2. Check configuration
print("\\n2️⃣ Configuration:")
config = get_config()
print(f"   Profile: {getattr(config, '_profile', 'default')}")
print(f"   Microlaw: {config.microlaw}")
print(f"   B(N) Family: {config.bn_family}")
print(f"   B(N) Params: {config.bn_params}")
print(f"   Validation: {'✅' if config.enable_validation else '❌'}")
print(f"   Stamping: {'✅' if config.enable_stamping else '❌'}")

# 3. Test computation
print("\\n3️⃣ Test Computation:")
try:
    inputs = MicrolawInput(phi=0.1, v=0.0)
    result = compute_BN(
        config.microlaw, inputs, config.microlaw_params,
        config.bn_family, config.bn_params
    )
    print(f"   Input: φ/c² = {inputs.phi}")
    print(f"   Output: N = {result['N']:.6f}, B = {result['B']:.6f}")
    print("   Status: ✅ Computation successful")
except Exception as e:
    print(f"   Status: ❌ Computation failed: {e}")

# 4. Test validation (if enabled)
if config.enable_validation:
    print("\\n4️⃣ Validation Check:")
    try:
        from tacc.core.microlaw import get_microlaw
        from tacc.core.bn import get_bn_family
        
        microlaw = get_microlaw(config.microlaw)
        bn_family = get_bn_family(config.bn_family)
        
        validation_result = validate_complete_setup(
            microlaw, inputs, config.microlaw_params,
            bn_family, config.bn_params
        )
        
        if validation_result["overall_valid"]:
            print("   Status: ✅ All validations passed")
        else:
            print("   Status: ⚠️ Some validations failed")
            for v in validation_result["validations"]:
                if not v["is_valid"]:
                    print(f"     - {v['component']}: {v['message']}")
                    
    except Exception as e:
        print(f"   Status: ❌ Validation failed: {e}")

# 5. Test stamping (if enabled) 
if config.enable_stamping:
    print("\\n5️⃣ Stamping Check:")
    try:
        temp_dir = Path("/tmp/tacc_sanity_check")
        temp_dir.mkdir(exist_ok=True)
        
        stamp_path = stamp_experiment(
            experiment_name="sanity_check",
            output_dir=temp_dir,
            microlaw_name=config.microlaw,
            microlaw_params=config.microlaw_params,
            bn_family_name=config.bn_family,
            bn_params=config.bn_params,
            fitment_name=config.fitment_name,
            fitment_params=config.fitment_params
        )
        
        print(f"   Stamp created: {stamp_path}")
        print("   Status: ✅ Stamping successful")
        
        # Clean up
        stamp_path.unlink()
        temp_dir.rmdir()
        
    except Exception as e:
        print(f"   Status: ❌ Stamping failed: {e}")

print("\\n🎉 Sanity check complete!")
print("\\nNext steps:")
print("• Use the component chooser above to select different configurations")
print("• Run compute_BN() with your chosen settings")  
print("• Create experiments with automatic stamping and validation")
'''


def display_system_info() -> None:
    """Display comprehensive system information in notebook."""
    from IPython.display import display, HTML
    
    info_html = f"""
<div style="
    border: 1px solid #ddd; border-radius: 8px; padding: 15px; 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; margin: 10px 0;
">
    <h2 style="margin-top: 0; text-align: center;">🧮 TACC System Information</h2>
    
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;">
        <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
            <h4>📊 Available Components</h4>
            <ul style="margin: 0; padding-left: 20px;">
                <li>Microlaws: {len(list_microlaws())}</li>
                <li>B(N) Families: {len(list_bn_families())}</li>
                <li>Fitments: {len(list_fitments())}</li>
                <li>Config Profiles: {len(list_profiles())}</li>
            </ul>
        </div>
        
        <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
            <h4>⚙️ Current Configuration</h4>
            <ul style="margin: 0; padding-left: 20px;">
                <li>Microlaw: {get_config().microlaw}</li>
                <li>B(N) Family: {get_config().bn_family}</li>
                <li>Fitment: {get_config().fitment_name}</li>
                <li>Validation: {'✅' if get_config().enable_validation else '❌'}</li>
            </ul>
        </div>
    </div>
    
    <div style="text-align: center; margin-top: 20px;">
        <em>Use the component chooser below to modify settings</em>
    </div>
</div>
"""
    
    display(HTML(info_html))
