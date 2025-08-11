"""
Example notebook cell demonstrating the fitment framework.
Copy and paste this into a Jupyter notebook cell to try it out.
"""

# Example notebook cell for fitment framework
from tacc.core.nb import list_fitments, get_fitment, compute_BN
from tacc.core.microlaw import MicrolawInput

# List available fitments
print("Available fitments:")
print(list_fitments())

# Example 1: No-fit (pass-through)
print("\n=== Example 1: No-fit ===")
fit = get_fitment("no_fit")
fitted = fit.fit([], "ppn", {}, "exponential", {"kappa": 2.0})
print(f"No-fit result: {fitted}")

# Example 2: Single parameter fitting
print("\n=== Example 2: Single parameter fitting ===")

# Create synthetic dataset
dataset = [
    {"x": {"phi": 0.0}, "target": 1.0, "weight": 1.0},
    {"x": {"phi": 0.1}, "target": 1.05, "weight": 1.0},
    {"x": {"phi": 0.2}, "target": 1.10, "weight": 1.0}
]

# Fit kappa parameter
fit = get_fitment("single_param")
fitted = fit.fit(
    dataset, 
    "ppn", 
    {}, 
    "exponential", 
    {"kappa": 2.0},
    extra={"target_param": "kappa", "init": 1.0, "max_iter": 20}
)

print(f"Fitted kappa: {fitted['bn']['kappa']:.3f}")

# Example 3: Use fitted parameters
print("\n=== Example 3: Using fitted parameters ===")
inputs = MicrolawInput(phi=0.1)
result = compute_BN(
    "ppn", 
    inputs, 
    fitted["microlaw"], 
    "exponential", 
    fitted["bn"]
)
print(f"With fitted kappa={fitted['bn']['kappa']:.3f}:")
print(f"  N = {result['N']:.3f}")
print(f"  B = {result['B']:.3f}")

print("\n✅ Fitment framework working correctly!")
