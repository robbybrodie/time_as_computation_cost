"""
Causal Diamond experiment: lightcone/diamond lattice construction and propagation.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any, Union
import json
import math


def run_demo(**kwargs) -> plt.Figure:
    """
    Create and visualize a causal diamond lattice with optional capacity gradient.
    
    Args:
        depth (int): Depth D of the diamond lattice (default: 10)
        alpha (float): Capacity gradient parameter (default: 0.1)
        show_front (bool): Whether to show propagation front (default: True)
        
    Returns:
        matplotlib.figure.Figure: The lattice visualization
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parse parameters
    depth = kwargs.get('depth', 10)
    alpha = kwargs.get('alpha', 0.1)
    show_front = kwargs.get('show_front', True)
    
    # Create diamond lattice
    G = create_diamond_lattice(depth)
    
    # Compute capacity values
    capacities = {}
    for node in G.nodes():
        t, x = node
        # N(t,x) = clip(1 - α*(|x|+t)/D, 0, 1)
        capacity = np.clip(1 - alpha * (abs(x) + t) / depth, 0, 1)
        capacities[node] = capacity
        
    # Set up figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Lattice structure with capacity coloring
    pos = {(t, x): (x, t) for t, x in G.nodes()}
    node_colors = [capacities[node] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          cmap='viridis', node_size=50, ax=ax1)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, ax=ax1)
    
    ax1.set_title(f'Causal Diamond Lattice (D={depth}, α={alpha})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label='Capacity N(t,x)')
    
    # Plot 2: Propagation front simulation if requested
    if show_front:
        front_data = simulate_propagation_front(G, capacities, depth)
        
        ax2.plot(front_data['times'], front_data['mean_positions'], 'b-', 
                linewidth=2, label='Mean |x|')
        ax2.fill_between(front_data['times'], 
                        front_data['mean_positions'] - front_data['std_positions'],
                        front_data['mean_positions'] + front_data['std_positions'],
                        alpha=0.3, label='± 1 std')
        
        # Theoretical expectation for symmetric random walk
        theoretical = np.sqrt(np.array(front_data['times']) * 0.5)  # sqrt(t/2) for random walk
        ax2.plot(front_data['times'], theoretical, 'r--', 
                label='Theoretical sqrt(t/2)')
        
        ax2.set_xlabel('Time t')
        ax2.set_ylabel('Mean |x| position')
        ax2.set_title('Propagation Front Profile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Front simulation disabled', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('No Front Simulation')
    
    plt.tight_layout()
    return fig


def run_experiment(**kwargs) -> Dict[str, Any]:
    """
    Run the complete causal diamond experiment and return metrics.
    
    Args:
        depth (int): Depth D of the diamond lattice (default: 10)
        alpha (float): Capacity gradient parameter (default: 0.1)
        output_dir (str): Directory to save outputs (default: None)
        
    Returns:
        dict: Experiment results containing metrics and file paths
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parse parameters
    depth = kwargs.get('depth', 10)
    alpha = kwargs.get('alpha', 0.1)
    output_dir = kwargs.get('output_dir', None)
    
    # Create diamond lattice
    G = create_diamond_lattice(depth)
    
    # Compute capacity values
    capacities = {}
    for node in G.nodes():
        t, x = node
        capacity = np.clip(1 - alpha * (abs(x) + t) / depth, 0, 1)
        capacities[node] = capacity
    
    # Basic metrics
    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    
    # Count paths from t=0 to each time layer
    num_paths_t = count_paths_by_time(G, depth)
    
    # Simulate front propagation
    front_data = simulate_propagation_front(G, capacities, depth)
    
    # Compute front profile metrics
    front_profile_t = front_data['mean_positions']
    front_symmetry_deviation = np.mean(np.abs(front_profile_t - np.sqrt(np.array(front_data['times']) * 0.5)))
    
    # Create results dictionary
    results = {
        'parameters': {
            'depth': depth,
            'alpha': alpha
        },
        'metrics': {
            'node_count': node_count,
            'edge_count': edge_count,
            'num_paths_t': num_paths_t,
            'front_profile_t': front_profile_t.tolist(),
            'front_symmetry_deviation': float(front_symmetry_deviation),
            'theoretical_paths_top': int(2**depth),  # Binomial coefficient approximation
            'actual_paths_top': num_paths_t[-1] if num_paths_t else 0
        },
        'front_data': {
            'times': front_data['times'],
            'mean_positions': front_data['mean_positions'].tolist(),
            'std_positions': front_data['std_positions'].tolist()
        }
    }
    
    # Save outputs if directory specified
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save demo plot
        fig = run_demo(depth=depth, alpha=alpha, show_front=True)
        demo_path = os.path.join(output_dir, 'diamond_demo.png')
        fig.savefig(demo_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        results['files'] = {'demo_plot': demo_path}
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        results['files']['metrics'] = metrics_path
    
    return results


def create_diamond_lattice(depth: int) -> nx.DiGraph:
    """
    Create a causal diamond lattice with nodes at (t,x) satisfying |x| ≤ t ≤ depth.
    Edges connect (t,x) → (t+1, x±1).
    
    Args:
        depth (int): Maximum time coordinate
        
    Returns:
        nx.DiGraph: Directed graph representing the diamond lattice
    """
    G = nx.DiGraph()
    
    # Add nodes: (t,x) where |x| <= t <= depth
    for t in range(depth + 1):
        for x in range(-t, t + 1):
            G.add_node((t, x))
    
    # Add edges: (t,x) → (t+1, x±1) if target node exists
    for t in range(depth):
        for x in range(-t, t + 1):
            # Forward light cone edges
            if abs(x - 1) <= t + 1:  # Check if (t+1, x-1) is valid
                G.add_edge((t, x), (t + 1, x - 1))
            if abs(x + 1) <= t + 1:  # Check if (t+1, x+1) is valid
                G.add_edge((t, x), (t + 1, x + 1))
    
    return G


def count_paths_by_time(G: nx.DiGraph, depth: int) -> list:
    """
    Count number of paths from t=0 to each time layer.
    
    Args:
        G (nx.DiGraph): Diamond lattice graph
        depth (int): Maximum time coordinate
        
    Returns:
        list: Number of paths to each time layer
    """
    # Start from the origin
    start_node = (0, 0)
    
    num_paths = []
    for t in range(depth + 1):
        if t == 0:
            num_paths.append(1)  # One path to origin
        else:
            # Count nodes at time t reachable from origin
            target_nodes = [(t, x) for x in range(-t, t + 1) if G.has_node((t, x))]
            paths_to_t = 0
            for target in target_nodes:
                if nx.has_path(G, start_node, target):
                    # Count simple paths (this is expensive but gives exact count)
                    try:
                        paths = list(nx.all_simple_paths(G, start_node, target))
                        paths_to_t += len(paths)
                    except:
                        # Fallback: just count if path exists
                        paths_to_t += 1
            num_paths.append(paths_to_t)
    
    return num_paths


def simulate_propagation_front(G: nx.DiGraph, capacities: dict, depth: int, 
                             n_samples: int = 100) -> Dict[str, np.ndarray]:
    """
    Simulate propagation front with capacity-weighted random walks.
    
    Args:
        G (nx.DiGraph): Diamond lattice graph
        capacities (dict): Node capacity values
        depth (int): Maximum time
        n_samples (int): Number of random walk samples
        
    Returns:
        dict: Front propagation data (times, mean_positions, std_positions)
    """
    times = list(range(depth + 1))
    positions_by_time = {t: [] for t in times}
    
    # Run multiple random walks
    for _ in range(n_samples):
        current_pos = (0, 0)  # Start at origin
        
        for t in range(depth):
            positions_by_time[t].append(abs(current_pos[1]))  # Record |x|
            
            # Get possible next positions
            successors = list(G.successors(current_pos))
            if not successors:
                break
                
            # Weight by capacity
            weights = [capacities[node] for node in successors]
            total_weight = sum(weights)
            
            if total_weight > 0:
                # Normalize probabilities
                probs = [w / total_weight for w in weights]
                # Choose next position
                next_pos = np.random.choice(len(successors), p=probs)
                current_pos = successors[next_pos]
            else:
                # If all capacities are zero, choose uniformly
                current_pos = np.random.choice(successors)
        
        # Record final position
        if current_pos[0] <= depth:
            positions_by_time[current_pos[0]].append(abs(current_pos[1]))
    
    # Compute statistics
    mean_positions = []
    std_positions = []
    
    for t in times:
        if positions_by_time[t]:
            mean_positions.append(np.mean(positions_by_time[t]))
            std_positions.append(np.std(positions_by_time[t]))
        else:
            mean_positions.append(0)
            std_positions.append(0)
    
    return {
        'times': times,
        'mean_positions': np.array(mean_positions),
        'std_positions': np.array(std_positions)
    }
