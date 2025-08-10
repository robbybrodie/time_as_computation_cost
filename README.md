# Time as Computation Cost

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robbybrodie/time_as_computation_cost/blob/main/notebooks/00_Run_All_Experiments.ipynb)

**TACC** (Time as Computation Cost) - A computational-capacity model of time dilation and gravitation.

## Theory Overview

This project explores a novel approach to understanding spacetime and gravity through computational capacity constraints. The core idea: **time dilation emerges from limitations in computational capacity**.

**Key Concepts:**
- Spacetime metric: `ds² = -N²c²dt² + [1/B(N)]dx²`
- Constitutive law: `B(N) = exp[-κ(1-N)]`
- Computational capacity `N` acts as an effective gravitational potential
- Parameter `κ` controls deviations from General Relativity
- When `κ = 2`, the model exactly recovers Einstein's GR

## Run in Colab - One Click Experiments

| Experiment | Notebook | Description |
|-----------|----------|-------------|
| **🚀 Run All Experiments** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robbybrodie/time_as_computation_cost/blob/main/notebooks/00_Run_All_Experiments.ipynb) | Complete experimental suite with all tests |
| **Bandgaps Fitting** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robbybrodie/time_as_computation_cost/blob/main/notebooks/bandgaps_colab.ipynb) | DoF law fitting and model comparison |
| **PPN Parameters** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robbybrodie/time_as_computation_cost/blob/main/notebooks/ppn_colab.ipynb) | Post-Newtonian parameter extraction |
| **Solar System Tests** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robbybrodie/time_as_computation_cost/blob/main/notebooks/geodesics_colab.ipynb) | Light bending, Shapiro delay, Mercury precession |

### Quick Start
1. Click any "Open in Colab" badge above
2. Run the setup cell to clone the repository and install dependencies
3. Execute experiment cells to see results
4. Explore interactive parameter controls and visualizations

## Project Structure

```
time_as_computation_cost/
├── src/tacc/                 # Main TACC package
│   ├── metric.py            # Spacetime metric construction
│   ├── constitutive.py      # B(N) constitutive law
│   ├── ppn.py               # Post-Newtonian parameter extraction
│   ├── geodesics.py         # Solar system tests
│   ├── baselines.py         # Model comparison tools
│   └── micro/               # Microphysical models
├── experiments/             # Experiment runners
│   ├── run_bandgaps.py     # DoF law fitting
│   ├── run_ppn.py          # PPN parameter analysis
│   └── run_geodesics.py    # Solar system tests
├── notebooks/               # Colab-ready notebooks
│   ├── 00_Run_All_Experiments.ipynb
│   ├── bandgaps_colab.ipynb
│   ├── ppn_colab.ipynb
│   └── geodesics_colab.ipynb
└── capacity-physics/        # Legacy structure (deprecated)
```

## Key Results

### 🧮 Theoretical Framework
- Time dilation emerges from computational capacity constraints
- Free parameter κ controls deviation from General Relativity  
- κ=2 gives exact agreement with Einstein's GR

### 📊 Experimental Validation
- **DoF Laws**: Successfully fits `DoF(N) = exp[-a(1-N)]` from synthetic data
- **PPN Parameters**: Extracts γ=κ/2, β=1 with GR limit at κ=2
- **Solar System**: Reproduces light bending (1.75"), Shapiro delay, Mercury precession (43.1"/century)

### 🔬 Physical Insights
- Computational capacity N acts as effective gravitational potential
- Framework provides alternative foundation for spacetime geometry
- Potentially testable deviations from GR in extreme conditions

## Installation & Local Development

```bash
# Clone repository
git clone https://github.com/robbybrodie/time_as_computation_cost.git
cd time_as_computation_cost

# Install package (editable)
pip install -e .

# Run individual experiments
python experiments/run_bandgaps.py
python experiments/run_ppn.py
python experiments/run_geodesics.py

# Results saved to experiments/out/
```

## Dependencies

- Python 3.8+
- NumPy, SciPy, Matplotlib
- Pandas (for data handling)
- PyYAML (for configuration)

All dependencies are automatically installed in Colab environments.

## Physical Motivation

The TACC framework addresses fundamental questions:
- Why does time dilate near massive objects?
- Could spacetime geometry emerge from information-theoretic constraints?
- What if computational limits underlie gravitational effects?

By modeling time dilation through computational capacity constraints, we explore whether the computational complexity of physical processes could be a more fundamental description than traditional geometric approaches.

## Future Directions

- **Cosmological Applications**: Dark energy and expansion
- **Quantum Gravity**: Information-theoretic foundations
- **Experimental Tests**: Precision measurements to constrain κ
- **Astrophysical Phenomena**: Black holes, neutron stars

## Contributing

We welcome contributions! Whether you're interested in:
- Running experiments with real astrophysical data
- Extending the theoretical framework
- Improving computational methods
- Adding new visualization tools

See individual notebook troubleshooting sections for technical support.

## Citation

If you use this work in research, please cite:
```
Time as Computation Cost: A computational-capacity model of spacetime
https://github.com/robbybrodie/time_as_computation_cost
```

---

*Explore how computational constraints might shape the fabric of spacetime itself.*
