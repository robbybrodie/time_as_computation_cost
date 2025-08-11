"""
Mode Crowding experiment: occupancy vs capacity with softmax distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
import json
import hashlib


def _generate_parameter_seed(K: int, N_max: float, N_min: float, n_points: int, utility_type: str) -> int:
    """
    Generate a reproducible seed based on experiment parameters.
    
    This ensures that:
    - Different parameter combinations get different seeds → different random utilities
    - Same parameter combinations always get the same seed → reproducible results
    - The seed is deterministic but varies meaningfully with parameters
    
    Args:
        K (int): Number of modes
        N_max (float): Maximum capacity
        N_min (float): Minimum capacity
        n_points (int): Number of capacity points
        utility_type (str): Type of utility generation
        
    Returns:
        int: Seed value for numpy.random.seed()
    """
    # Create a string representation of parameters with sufficient precision
    param_string = f"crowding_K_{K}_Nmax_{N_max:.8f}_Nmin_{N_min:.8f}_n_{n_points}_type_{utility_type}"
    
    # Generate hash from parameters
    hash_obj = hashlib.md5(param_string.encode())
    
    # Convert first 4 bytes of hash to integer for seed
    seed = int.from_bytes(hash_obj.digest()[:4], byteorder='big')
    
    # Keep within valid numpy random seed range (0 to 2^32-1)
    seed = seed % (2**32)
    
    return seed


def run_demo(**kwargs) -> plt.Figure:
    """
    Create and visualize mode crowding behavior with capacity variation.
    
    Args:
        K (int): Number of modes (default: 10)
        N_max (float): Maximum capacity (default: 1.0)
        N_min (float): Minimum capacity (default: 0.01)
        n_points (int): Number of capacity points (default: 100)
        utility_type (str): 'random' or 'spaced' utilities (default: 'random')
        
    Returns:
        matplotlib.figure.Figure: The mode crowding visualization
    """
    # Parse parameters
    K = kwargs.get('K', 10)
    N_max = kwargs.get('N_max', 1.0)
    N_min = kwargs.get('N_min', 0.01)
    n_points = kwargs.get('n_points', 100)
    utility_type = kwargs.get('utility_type', 'random')
    
    # Set parameter-dependent random seed for reproducible but parameter-specific results
    seed = _generate_parameter_seed(K, N_max, N_min, n_points, utility_type)
    np.random.seed(seed)
    
    # Generate baseline utilities
    utilities = generate_utilities(K, utility_type)
    
    # Create capacity range
    N_values = np.linspace(N_max, N_min, n_points)
    
    # Compute metrics
    results = compute_crowding_metrics(utilities, N_values)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Participation Ratio vs N
    ax1.plot(N_values, results['participation_ratio'], 'b-', linewidth=2, label='PR(N)')
    ax1.axhline(y=results['PR_min'], color='r', linestyle='--', alpha=0.7, 
                label=f'PR_min = {results["PR_min"]:.2f}')
    if 'N_c_pr' in results:
        ax1.axvline(x=results['N_c_pr'], color='g', linestyle=':', alpha=0.7,
                    label=f'N_c (PR) = {results["N_c_pr"]:.3f}')
    ax1.set_xlabel('Capacity N')
    ax1.set_ylabel('Participation Ratio')
    ax1.set_title('Participation Ratio vs Capacity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(N_min, N_max)
    
    # Plot 2: Gini coefficient vs N
    ax2.plot(N_values, results['gini_coefficient'], 'r-', linewidth=2, label='Gini(N)')
    ax2.axhline(y=results['Gini_max'], color='b', linestyle='--', alpha=0.7,
                label=f'Gini_max = {results["Gini_max"]:.2f}')
    if 'N_c_gini' in results:
        ax2.axvline(x=results['N_c_gini'], color='g', linestyle=':', alpha=0.7,
                    label=f'N_c (Gini) = {results["N_c_gini"]:.3f}')
    ax2.set_xlabel('Capacity N')
    ax2.set_ylabel('Gini Coefficient')
    ax2.set_title('Inequality (Gini) vs Capacity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(N_min, N_max)
    
    # Plot 3: Mode occupancy heatmap
    occupancy_matrix = results['occupancy_matrix']
    # Show every 10th capacity point for clarity
    step = max(1, n_points // 20)
    im = ax3.imshow(occupancy_matrix[::step, :].T, aspect='auto', cmap='viridis',
                    extent=[N_min, N_max, 0, K])
    ax3.set_xlabel('Capacity N')
    ax3.set_ylabel('Mode Index')
    ax3.set_title('Mode Occupancy Heatmap')
    plt.colorbar(im, ax=ax3, label='Occupancy p_i')
    
    # Plot 4: Entropy vs N
    ax4.plot(N_values, results['entropy'], 'g-', linewidth=2, label='H(N)')
    ax4.axhline(y=np.min(results['entropy']), color='r', linestyle='--', alpha=0.7,
                label=f'H_min = {np.min(results["entropy"]):.2f}')
    ax4.set_xlabel('Capacity N')
    ax4.set_ylabel('Shannon Entropy')
    ax4.set_title('Entropy vs Capacity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(N_min, N_max)
    
    plt.tight_layout()
    return fig


def run_experiment(**kwargs) -> Dict[str, Any]:
    """
    Run the complete mode crowding experiment and return metrics.
    
    Args:
        K (int): Number of modes (default: 10)
        N_max (float): Maximum capacity (default: 1.0)
        N_min (float): Minimum capacity (default: 0.01)
        n_points (int): Number of capacity points (default: 100)
        utility_type (str): 'random' or 'spaced' utilities (default: 'random')
        threshold (float): Threshold for critical point detection (default: 0.1)
        output_dir (str): Directory to save outputs (default: None)
        
    Returns:
        dict: Experiment results containing critical points and metrics
    """
    # Parse parameters
    K = kwargs.get('K', 10)
    N_max = kwargs.get('N_max', 1.0)
    N_min = kwargs.get('N_min', 0.01)
    n_points = kwargs.get('n_points', 100)
    utility_type = kwargs.get('utility_type', 'random')
    threshold = kwargs.get('threshold', 0.1)
    output_dir = kwargs.get('output_dir', None)
    
    # Set parameter-dependent random seed for reproducible but parameter-specific results
    seed = _generate_parameter_seed(K, N_max, N_min, n_points, utility_type)
    np.random.seed(seed)
    
    # Generate baseline utilities
    utilities = generate_utilities(K, utility_type)
    
    # Create capacity range
    N_values = np.linspace(N_max, N_min, n_points)
    
    # Compute metrics
    crowding_results = compute_crowding_metrics(utilities, N_values)
    
    # Find critical points
    critical_points = find_critical_points(N_values, crowding_results, threshold)
    
    # Create results dictionary
    results = {
        'parameters': {
            'K': K,
            'N_max': N_max,
            'N_min': N_min,
            'n_points': n_points,
            'utility_type': utility_type,
            'threshold': threshold
        },
        'utilities': utilities.tolist(),
        'metrics': {
            'N_c_pr': critical_points.get('N_c_pr', None),
            'N_c_gini': critical_points.get('N_c_gini', None),
            'N_c_entropy': critical_points.get('N_c_entropy', None),
            'PR_min': crowding_results['PR_min'],
            'Gini_max': crowding_results['Gini_max'],
            'entropy_min': float(np.min(crowding_results['entropy'])),
            'entropy_max': float(np.max(crowding_results['entropy']))
        },
        'curves': {
            'N_values': N_values.tolist(),
            'participation_ratio': crowding_results['participation_ratio'].tolist(),
            'gini_coefficient': crowding_results['gini_coefficient'].tolist(),
            'entropy': crowding_results['entropy'].tolist()
        }
    }
    
    # Add critical point information
    results['critical_analysis'] = critical_points
    
    # Save outputs if directory specified
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save demo plot
        fig = run_demo(K=K, N_max=N_max, N_min=N_min, n_points=n_points, 
                      utility_type=utility_type)
        plot_path = os.path.join(output_dir, 'crowding_curves.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Save results
        results_path = os.path.join(output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        results['files'] = {
            'plot': plot_path,
            'results': results_path
        }
    
    return results


def generate_utilities(K: int, utility_type: str = 'random') -> np.ndarray:
    """
    Generate baseline utilities for K modes.
    
    Args:
        K (int): Number of modes
        utility_type (str): 'random' or 'spaced'
        
    Returns:
        np.ndarray: Utility values u_i
    """
    if utility_type == 'random':
        # Random utilities from standard normal
        utilities = np.random.normal(0, 1, K)
    elif utility_type == 'spaced':
        # Linearly spaced utilities
        utilities = np.linspace(0, 2, K)
    else:
        raise ValueError(f"Unknown utility_type: {utility_type}")
    
    # Sort in descending order for clarity
    utilities = np.sort(utilities)[::-1]
    return utilities


def softmax_occupancy(utilities: np.ndarray, N: float) -> np.ndarray:
    """
    Compute occupancy distribution via softmax with temperature = N.
    
    Args:
        utilities (np.ndarray): Baseline utilities u_i
        N (float): Capacity parameter (temperature)
        
    Returns:
        np.ndarray: Occupancy probabilities p_i
    """
    # Temperature with clipping to avoid numerical issues
    tau = np.clip(N, 1e-3, 1.0)
    
    # Softmax distribution
    exp_values = np.exp(utilities / tau)
    probabilities = exp_values / np.sum(exp_values)
    
    return probabilities


def participation_ratio(probabilities: np.ndarray) -> float:
    """
    Compute participation ratio: PR = 1 / sum(p_i^2).
    
    Args:
        probabilities (np.ndarray): Mode occupancy probabilities
        
    Returns:
        float: Participation ratio (effective number of active modes)
    """
    return 1.0 / np.sum(probabilities**2)


def gini_coefficient(probabilities: np.ndarray) -> float:
    """
    Compute Gini coefficient for inequality measure.
    
    Args:
        probabilities (np.ndarray): Mode occupancy probabilities
        
    Returns:
        float: Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    # Sort probabilities
    sorted_p = np.sort(probabilities)
    n = len(sorted_p)
    
    # Compute Gini coefficient
    cumsum = np.cumsum(sorted_p)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_p))) / (n * cumsum[-1]) - (n + 1) / n
    
    return gini


def shannon_entropy(probabilities: np.ndarray) -> float:
    """
    Compute Shannon entropy: H = -sum(p_i * log(p_i)).
    
    Args:
        probabilities (np.ndarray): Mode occupancy probabilities
        
    Returns:
        float: Shannon entropy
    """
    # Avoid log(0) by adding small epsilon
    p_safe = probabilities + 1e-12
    return -np.sum(probabilities * np.log(p_safe))


def compute_crowding_metrics(utilities: np.ndarray, N_values: np.ndarray) -> Dict[str, Any]:
    """
    Compute all crowding metrics for range of capacity values.
    
    Args:
        utilities (np.ndarray): Baseline utilities
        N_values (np.ndarray): Range of capacity values
        
    Returns:
        dict: All computed metrics and curves
    """
    n_points = len(N_values)
    K = len(utilities)
    
    # Initialize arrays
    participation_ratios = np.zeros(n_points)
    gini_coefficients = np.zeros(n_points)
    entropies = np.zeros(n_points)
    occupancy_matrix = np.zeros((n_points, K))
    
    # Compute metrics for each capacity value
    for i, N in enumerate(N_values):
        probabilities = softmax_occupancy(utilities, N)
        
        participation_ratios[i] = participation_ratio(probabilities)
        gini_coefficients[i] = gini_coefficient(probabilities)
        entropies[i] = shannon_entropy(probabilities)
        occupancy_matrix[i, :] = probabilities
    
    return {
        'participation_ratio': participation_ratios,
        'gini_coefficient': gini_coefficients,
        'entropy': entropies,
        'occupancy_matrix': occupancy_matrix,
        'PR_min': float(np.min(participation_ratios)),
        'Gini_max': float(np.max(gini_coefficients))
    }


def find_critical_points(N_values: np.ndarray, results: Dict[str, np.ndarray], 
                        threshold: float = 0.1) -> Dict[str, Any]:
    """
    Find critical capacity points where derivatives cross threshold.
    
    Args:
        N_values (np.ndarray): Capacity values
        results (dict): Computed metrics
        threshold (float): Threshold for derivative detection
        
    Returns:
        dict: Critical point analysis
    """
    critical_points = {}
    
    # Compute numerical derivatives
    dN = N_values[1] - N_values[0]  # Assuming uniform spacing
    
    # Participation ratio derivative
    dPR_dN = np.gradient(results['participation_ratio'], dN)
    critical_points['dPR_dN'] = dPR_dN.tolist()
    
    # Find where |dPR/dN| exceeds threshold
    high_derivative_mask = np.abs(dPR_dN) > threshold
    if np.any(high_derivative_mask):
        critical_idx = np.argmax(np.abs(dPR_dN))
        critical_points['N_c_pr'] = float(N_values[critical_idx])
        critical_points['max_dPR_dN'] = float(np.abs(dPR_dN[critical_idx]))
    
    # Gini coefficient derivative
    dGini_dN = np.gradient(results['gini_coefficient'], dN)
    critical_points['dGini_dN'] = dGini_dN.tolist()
    
    # Find where |dGini/dN| exceeds threshold
    high_derivative_mask_gini = np.abs(dGini_dN) > threshold
    if np.any(high_derivative_mask_gini):
        critical_idx = np.argmax(np.abs(dGini_dN))
        critical_points['N_c_gini'] = float(N_values[critical_idx])
        critical_points['max_dGini_dN'] = float(np.abs(dGini_dN[critical_idx]))
    
    # Entropy derivative
    dH_dN = np.gradient(results['entropy'], dN)
    critical_points['dH_dN'] = dH_dN.tolist()
    
    # Find where |dH/dN| exceeds threshold
    high_derivative_mask_entropy = np.abs(dH_dN) > threshold
    if np.any(high_derivative_mask_entropy):
        critical_idx = np.argmax(np.abs(dH_dN))
        critical_points['N_c_entropy'] = float(N_values[critical_idx])
        critical_points['max_dH_dN'] = float(np.abs(dH_dN[critical_idx]))
    
    # Summary statistics
    critical_points['threshold_used'] = threshold
    critical_points['has_pr_critical'] = 'N_c_pr' in critical_points
    critical_points['has_gini_critical'] = 'N_c_gini' in critical_points
    critical_points['has_entropy_critical'] = 'N_c_entropy' in critical_points
    
    return critical_points
