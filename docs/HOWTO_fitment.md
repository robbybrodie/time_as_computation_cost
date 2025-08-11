# Fitment Framework HOWTO

This guide explains how to use the TACC fitment framework for calibrating microlaws and B(N) families.

## What are Fitments?

Fitments provide a pluggable calibration framework that cleanly separates:

- **Microlaws** (physics: invariants → N)
- **B(N) families** (response: N → geometry factor) 
- **Fitments** (optional calibration producing params for microlaws and/or B(N))

The framework works seamlessly from scripts, CLI, and Jupyter notebooks without touching experiment logic.

## Available Fitments

### no_fit
Pass-through fitment that returns parameters unchanged. Useful for testing or when no calibration is needed.

### single_param
1D optimizer for fitting a single scalar parameter (e.g., kappa or eta). Uses parabolic step optimization.

## Using Fitments in Notebooks

### Basic Setup

```python
from tacc.core.nb import list_fitments, get_fitment, compute_BN
from tacc.core.microlaw import MicrolawInput

# List available fitments
print(list_fitments())

# Get a specific fitment
fit = get_fitment("no_fit")
```

### Example: No-fit (Pass-through)

```python
# Create test inputs
inputs = MicrolawInput(phi=0.1)  # Weak gravitational field

# Define initial parameters
MICROLAW = "ppn"
ML_PARAMS = {}
BN_FAMILY = "exponential"
BN_PARAMS = {"kappa": 2.0}

# Apply no-fit (returns params unchanged)
fit = get_fitment("no_fit")
fitted = fit.fit([], MICROLAW, ML_PARAMS, BN_FAMILY, BN_PARAMS)

# Update parameters (no change expected)
ML_PARAMS.update(fitted["microlaw"])
BN_PARAMS.update(fitted["bn"])

# Compute result
result = compute_BN(MICROLAW, inputs, ML_PARAMS, BN_FAMILY, BN_PARAMS)
print(f"N = {result['N']:.3f}, B = {result['B']:.3f}")
```

### Example: Single Parameter Fitting

```python
# Create synthetic dataset
dataset = [
    {
        "x": {"phi": 0.0},  # MicrolawInput kwargs
        "target": 1.0,      # Expected B value
        "weight": 1.0       # Optional weight
    },
    {
        "x": {"phi": 0.1},
        "target": 1.05,
        "weight": 1.0
    },
    {
        "x": {"phi": 0.2},
        "target": 1.10,
        "weight": 1.0
    }
]

# Apply single parameter fitting
fit = get_fitment("single_param")
fitted = fit.fit(
    dataset, 
    MICROLAW, 
    ML_PARAMS, 
    BN_FAMILY, 
    BN_PARAMS,
    loss=None,  # Uses default squared error
    extra={
        "target_param": "kappa",  # Parameter to optimize
        "init": 2.0,              # Initial guess
        "max_iter": 30,           # Maximum iterations
        "tol": 1e-6               # Convergence tolerance
    }
)

# Update parameters with fitted values
ML_PARAMS.update(fitted["microlaw"])
BN_PARAMS.update(fitted["bn"])
print(f"Fitted kappa: {BN_PARAMS['kappa']:.3f}")
```

### Custom Loss Function

```python
def custom_loss(data_row, prediction):
    """Custom loss function example."""
    target = data_row["target"]
    pred_B = prediction["B"]
    return abs(pred_B - target)  # L1 loss instead of L2

fitted = fit.fit(
    dataset, MICROLAW, ML_PARAMS, BN_FAMILY, BN_PARAMS,
    loss=custom_loss,
    extra={"target_param": "kappa", "init": 1.5}
)
```

## Using Fitments from CLI

### Basic Usage

```bash
# List available components
python -m tacc.cli.main list

# Run with no_fit (default)
python -m tacc.cli.main dummy --fitment no_fit

# Run PPN experiment with no fitting
python -m tacc.cli.main ppn
```

### Single Parameter Fitting

```bash
# Fit kappa parameter using single_param fitment
python -m tacc.cli.main ppn \
    --fitment single_param \
    --fitment-params '{"target_param":"kappa","init":2.0,"max_iter":50}' \
    --dataset data/ppn_sample.json
```

### Custom Configuration

```bash
# Use custom config file
python -m tacc.cli.main ppn --config my_config.yaml

# Override config parameters
python -m tacc.cli.main ppn \
    --microlaw ppn \
    --bn-family exponential \
    --fitment single_param
```

## Dataset Format

Datasets should be JSON files containing arrays of data points:

```json
[
    {
        "x": {"phi": 0.0, "v": 0.1},
        "target": 1.0,
        "weight": 1.0
    },
    {
        "x": {"phi": 0.1, "v": 0.0},
        "target": 1.05,
        "weight": 2.0
    }
]
```

Each data point must have:
- `x`: Dictionary of MicrolawInput parameters
- `target`: Expected output value (typically B)
- `weight`: Optional weight for loss function (default 1.0)

## Configuration Files

YAML configuration files can specify default parameters:

```yaml
# default.yaml
microlaw: ppn
microlaw_params: {}

bn_family: exponential
bn_params:
  kappa: 2.0

fitment:
  name: single_param
  params:
    target_param: kappa
    init: 1.8
    max_iter: 50
    tol: 1e-6
```

## Advanced Usage

### Fitting Microlaw Parameters

```python
# Fit a parameter that lives in the microlaw
ML_PARAMS = {"eta": 1.0}  # Add microlaw parameter
BN_PARAMS = {"kappa": 2.0}

fitted = fit.fit(
    dataset, MICROLAW, ML_PARAMS, BN_FAMILY, BN_PARAMS,
    extra={"target_param": "eta", "init": 1.5}  # Will update ML_PARAMS
)
```

### Multiple Datasets

```python
# Combine multiple datasets
dataset1 = load_ppn_data()
dataset2 = load_sr_data()
combined_dataset = list(dataset1) + list(dataset2)

fitted = fit.fit(combined_dataset, ...)
```

## Tips and Best Practices

1. **Start Simple**: Begin with `no_fit` to verify your pipeline works
2. **Check Convergence**: Monitor the `max_iter` parameter for `single_param`
3. **Validate Results**: Always check that fitted parameters make physical sense
4. **Use Weights**: Weight important data points higher in your dataset
5. **Custom Loss**: Implement domain-specific loss functions when needed

## Troubleshooting

### Common Issues

- **Import Errors**: Ensure `import tacc.fitments` to register fitments
- **Dataset Format**: Verify JSON structure matches expected format
- **Convergence**: Increase `max_iter` or adjust `tol` for difficult fits
- **Parameter Location**: Check if parameter exists in microlaw or B(N) family

### Debug Mode

Use smaller datasets and more verbose output for debugging:

```python
# Simple test case
test_dataset = [{"x": {"phi": 0.0}, "target": 1.0, "weight": 1.0}]
result = fit.fit(test_dataset, ...)
print(f"Fitment result: {result}")
