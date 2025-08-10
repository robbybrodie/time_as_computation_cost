# TACC Experiment Rationales

**Why does each experiment exist? What role does it play in testing the TACC framework?**

This document provides plain-English explanations for every experiment in the Time as Computational Cost repository, helping researchers understand the purpose and significance of each test.

## Overview

The TACC experimental program spans four physical scales, from microscopic computational foundations to cosmological applications. Each experiment serves a specific role in validating or exploring different aspects of the computational-capacity approach to spacetime geometry.

## Experiment Categories

### Foundation Tests (Microscopic Scale)
These experiments test whether the basic computational building blocks work correctly.

### Robustness Tests (Microscopic Scale) 
These experiments test whether the framework can handle real-world conditions like noise and model uncertainty.

### Scaling Tests (Mesoscopic Scale)
These experiments test whether computational capacity follows expected statistical mechanical behavior.

### Validation Tests (Macroscopic Scale)
These experiments test whether TACC reproduces known physics in well-understood limits.

### Application Tests (Cosmological & Astrophysical Scale)
These experiments test whether TACC works at the largest scales and most extreme conditions.

---

## Individual Experiment Rationales

### 1. Causal Diamond (Foundation Test)

**Physical Scale:** Microscopic  
**What It Tests:** Lightcone lattice structure and computational node connectivity  
**Why It's Essential:** Foundation test - Can computational nodes form consistent spacetime geometry? Like checking if your building blocks fit together before constructing the theory.  
**Expected Result:** Symmetric lightcones with proper causal structure  
**Key Observable:** Lightcone geometry  
**TACC Parameter:** Lattice structure  

**Analogy:** This is like testing whether LEGO blocks actually connect properly before trying to build a castle.

---

### 2. Tension Bandgaps (Robustness Test)

**Physical Scale:** Microscopic  
**What It Tests:** Parameter recovery in synthetic data with noise and model selection  
**Why It's Essential:** Robustness test - Can we reliably extract TACC parameters from noisy data? Essential for real-world applications where data is never perfect.  
**Expected Result:** Accurate recovery of α, β parameters despite noise  
**Key Observable:** Capacity constraints  
**TACC Parameter:** α, β coefficients  

**Analogy:** This is like testing whether you can still read a road sign when it's raining - if your theory breaks down with a little noise, it's not practical.

---

### 3. Mode Crowding (Phase Transition Test)

**Physical Scale:** Mesoscopic  
**What It Tests:** Critical point behavior when computational modes become occupied  
**Why It's Essential:** Phase transition test - Does the model exhibit realistic critical behavior? Tests if computational "traffic jams" create observable effects.  
**Expected Result:** Critical point at finite occupancy ratio  
**Key Observable:** Mode distribution  
**TACC Parameter:** Occupancy ratio  

**Analogy:** This tests whether computational "traffic jams" create measurable effects, like how highway congestion suddenly appears at critical traffic density.

---

### 4. Bandgaps DoF (Scaling Test)

**Physical Scale:** Mesoscopic  
**What It Tests:** Degrees of freedom scaling laws with computational capacity  
**Why It's Essential:** Scaling test - Does computational capacity follow expected statistical mechanics? Validates connection to thermodynamics.  
**Expected Result:** DoF ∝ exp[-α(1-N)] scaling law  
**Key Observable:** Degrees of freedom  
**TACC Parameter:** DoF scaling law  

**Analogy:** This checks whether computational capacity behaves like a thermodynamic quantity - if it doesn't scale properly, the whole statistical mechanical foundation is questionable.

---

### 5. PPN Parameters (Sanity Check Test)

**Physical Scale:** Macroscopic  
**What It Tests:** Post-Newtonian parameter extraction (γ, β) from metric  
**Why It's Essential:** Weak-field test - Does TACC reduce to known physics in familiar limits? The "sanity check" - if PPN fails, the model is dead on arrival.  
**Expected Result:** γ = κ/2, β = 1, with κ ≈ 2 for GR limit  
**Key Observable:** γ, β PPN parameters  
**TACC Parameter:** κ parameter  

**Analogy:** This is like making sure your ruler measures a meter as a meter before building a bridge. PPN parameters are the "ruler" for testing gravity theories.

---

### 6. Solar System Tests (Historical Validation)

**Physical Scale:** Macroscopic  
**What It Tests:** Light bending, Shapiro delay, Mercury precession  
**Why It's Essential:** Historical validation - Can TACC reproduce the classic tests that made Einstein famous? These are precision, well-measured effects.  
**Expected Result:** 1.75" deflection, ~100μs delays, 43"/century precession  
**Key Observable:** Precession, deflection  
**TACC Parameter:** κ ≈ 2.0  

**Analogy:** These are the "final exams" that every gravity theory must pass. Einstein passed them, Newton failed them. Where does TACC stand?

---

### 7. Cosmological Expansion (Large-Scale Test)

**Physical Scale:** Cosmological  
**What It Tests:** FLRW metric consistency and expansion history H(z)  
**Why It's Essential:** Large-scale test - Does TACC work for the entire universe, not just local gravity? Bridge between micro and macro physics.  
**Expected Result:** Dark energy emerges from computational constraints  
**Key Observable:** Distance modulus  
**TACC Parameter:** κ constitutive  

**Analogy:** This tests whether your local building blocks can construct an entire city. Just because something works in your neighborhood doesn't mean it scales up.

---

### 8. Black Hole Thermodynamics (Extreme Gravity Test)

**Physical Scale:** Astrophysical  
**What It Tests:** Hawking temperature and entropy scaling with computational capacity  
**Why It's Essential:** Extreme gravity test - Can TACC handle the most extreme spacetime conditions? Tests information-theoretic connections.  
**Expected Result:** Modified Hawking radiation with κ dependence  
**Key Observable:** Hawking temperature  
**TACC Parameter:** κ entropy relation  

**Analogy:** This is the "stress test" - if your theory can handle black holes (the most extreme environments in the universe), it's probably robust.

---

### 9. Gravitational Waves (Dynamic Spacetime Test)

**Physical Scale:** Astrophysical  
**What It Tests:** Wave propagation speed and phase evolution in TACC metric  
**Why It's Essential:** Dynamic spacetime test - Does TACC predict correct wave behavior? Critical for multi-messenger astronomy.  
**Expected Result:** Wave speed modifications and phase shifts  
**Key Observable:** GW phase evolution  
**TACC Parameter:** κ wave speed  

**Analogy:** This tests whether spacetime can "wiggle" correctly in your theory. It's not enough to get static configurations right - dynamics must work too.

---

## Experimental Logic Flow

The experiments are designed to build confidence progressively:

1. **Foundation** → Do the basic building blocks work?
2. **Robustness** → Can we extract parameters reliably?  
3. **Scaling** → Does the statistical mechanics make sense?
4. **Validation** → Does it reproduce known physics?
5. **Application** → Does it work at all scales and conditions?

## Critical Decision Points

- **If Foundation Tests Fail:** The entire approach is flawed
- **If Robustness Tests Fail:** The framework isn't practical  
- **If Scaling Tests Fail:** The thermodynamics is wrong
- **If Validation Tests Fail:** The model is ruled out by data
- **If Application Tests Fail:** The theory doesn't generalize

## Key Insights Across All Experiments

1. **Parameter κ** emerges as the central quantity controlling deviations from General Relativity
2. **Computational capacity N** acts as an effective gravitational potential
3. **Scale hierarchy** allows testing from quantum-scale processes to cosmic evolution
4. **Multiple validation paths** provide redundant checks against known physics
5. **Observable predictions** distinguish TACC from other alternative gravity theories

---

## For Researchers

**Using These Rationales:**
- Include relevant rationales at the top of individual notebooks
- Reference this document when explaining experimental design
- Use the analogies to communicate with non-specialist audiences
- Cite the progressive logic when justifying experimental sequences

**Contributing New Experiments:**
- Identify which category your experiment fits into
- Explain why it's essential (what gap does it fill?)
- Define clear expected results and key observables
- Connect to the overall experimental logic flow

---

*These rationales help researchers quickly understand not just what each experiment does, but why it matters for the overall TACC research program.*
