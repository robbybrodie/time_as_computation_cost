"""
Tension Bandgaps experiment: micro fit + model selection with synthetic data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from typing import Dict, Any, List, Tuple
import json
import hashlib


def _generate_parameter_seed(n_points: int, noise_sigma: float, a_true: float, beta_true: float) -> int:
    """
    Generate a reproducible seed based on experiment parameters.
    
    This ensures that:
    - Different parameter combinations get different seeds → different synthetic data
    - Same parameter combinations always get the same seed → reproducible results
    - The seed is deterministic but varies meaningfully with parameters
    
    Args:
        n_points (int): Number of data points
        noise_sigma (float): Noise level 
        a_true (float): True exponential parameter
        beta_true (float): True power law parameter
        
    Returns:
        int: Seed value for numpy.random.seed()
    """
    # Create a string representation of parameters with sufficient precision
    param_string = f"bandgaps_n_{n_points}_sigma_{noise_sigma:.8f}_a_{a_true:.8f}_beta_{beta_true:.8f}"
    
    # Generate hash from parameters
    hash_obj = hashlib.md5(param_string.encode())
    
    # Convert first 4 bytes of hash to integer for seed
    seed = int.from_bytes(hash_obj.digest()[:4], byteorder='big')
    
    # Keep within valid numpy random seed range (0 to 2^32-1)
    seed = seed % (2**32)
    
    return seed


def run_demo(**kwargs) -> plt.Figure:
    """
    Create and visualize tension bandgaps fitting with model comparison.
    
    Args:
        n_points (int): Number of data points (default: 50)
        noise_sigma (float): Gaussian noise level (default: 0.05)
        a_true (float): True exponential parameter (default: 2.0)
        beta_true (float): True power law parameter (default: 1.5)
        
    Returns:
        matplotlib.figure.Figure: The fitting visualization
    """
    # Parse parameters
    n_points = kwargs.get('n_points', 50)
    noise_sigma = kwargs.get('noise_sigma', 0.05)
    a_true = kwargs.get('a_true', 2.0)
    beta_true = kwargs.get('beta_true', 1.5)
    
    # Set parameter-dependent random seed for reproducible but parameter-specific results
    seed = _generate_parameter_seed(n_points, noise_sigma, a_true, beta_true)
    np.random.seed(seed)
    
    # Generate synthetic data
    data = generate_synthetic_data(n_points, noise_sigma, a_true, beta_true)
    
    # Fit models
    models = fit_all_models(data['N_values'], data['DoF_values'], data['psi_values'])
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: DoF fitting
    N_fine = np.linspace(0.7, 1.0, 200)
    
    ax1.scatter(data['N_values'], data['DoF_values'], alpha=0.6, label='Data', s=20)
    
    # Plot fitted models
    DoF_exp = exponential_model(N_fine, models['exponential']['params'][0])
    ax1.plot(N_fine, DoF_exp, 'r-', linewidth=2, label=f"Exponential (a={models['exponential']['params'][0]:.2f})")
    
    DoF_poly = polynomial_model(N_fine, *models['polynomial']['params'])
    ax1.plot(N_fine, DoF_poly, 'g--', linewidth=2, label="Polynomial-2")
    
    DoF_power = power_law_model(N_fine, *models['power_law']['params'])
    ax1.plot(N_fine, DoF_power, 'b:', linewidth=2, label="Power Law")
    
    # True model
    DoF_true = exponential_model(N_fine, a_true)
    ax1.plot(N_fine, DoF_true, 'k-', alpha=0.5, linewidth=1, label=f"True (a={a_true})")
    
    ax1.set_xlabel('N (Computational Capacity)')
    ax1.set_ylabel('DoF (Degrees of Freedom)')
    ax1.set_title('DoF Model Fitting')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Psi fitting
    DoF_fine = np.linspace(0.5, 2.0, 200)
    
    ax2.scatter(data['DoF_values'], data['psi_values'], alpha=0.6, label='Data', s=20)
    
    # Plot fitted psi model
    psi_fit = DoF_fine ** models['psi_beta']['params'][0]
    ax2.plot(DoF_fine, psi_fit, 'r-', linewidth=2, 
             label=f"psi = DoF^β (β={models['psi_beta']['params'][0]:.2f})")
    
    # True psi model
    psi_true = DoF_fine ** beta_true
    ax2.plot(DoF_fine, psi_true, 'k-', alpha=0.5, linewidth=1, 
             label=f"True (β={beta_true})")
    
    ax2.set_xlabel('DoF')
    ax2.set_ylabel('ψ')
    ax2.set_title('ψ = DoF^β Fitting')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    residuals_exp = data['DoF_values'] - exponential_model(data['N_values'], models['exponential']['params'][0])
    residuals_poly = data['DoF_values'] - polynomial_model(data['N_values'], *models['polynomial']['params'])
    residuals_power = data['DoF_values'] - power_law_model(data['N_values'], *models['power_law']['params'])
    
    ax3.scatter(data['N_values'], residuals_exp, alpha=0.6, label='Exponential', s=20)
    ax3.scatter(data['N_values'], residuals_poly, alpha=0.6, label='Polynomial', s=20)
    ax3.scatter(data['N_values'], residuals_power, alpha=0.6, label='Power Law', s=20)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax3.set_xlabel('N')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Model Residuals')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model comparison
    model_names = ['Exponential', 'Polynomial', 'Power Law']
    aic_values = [models['exponential']['aic'], models['polynomial']['aic'], models['power_law']['aic']]
    bic_values = [models['exponential']['bic'], models['polynomial']['bic'], models['power_law']['bic']]
    cv_scores = [models['exponential']['cv_score'], models['polynomial']['cv_score'], models['power_law']['cv_score']]
    
    x_pos = np.arange(len(model_names))
    width = 0.25
    
    ax4.bar(x_pos - width, aic_values, width, label='AIC', alpha=0.8)
    ax4.bar(x_pos, bic_values, width, label='BIC', alpha=0.8)
    ax4.bar(x_pos + width, np.array(cv_scores) * 100, width, label='CV MSE (×100)', alpha=0.8)
    
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Score')
    ax4.set_title('Model Comparison (Lower is Better)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(model_names, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def run_experiment(**kwargs) -> Dict[str, Any]:
    """
    Run the complete tension bandgaps experiment and return results.
    
    Args:
        n_points (int): Number of data points (default: 50)
        noise_sigma (float): Gaussian noise level (default: 0.05)
        a_true (float): True exponential parameter (default: 2.0)
        beta_true (float): True power law parameter (default: 1.5)
        output_dir (str): Directory to save outputs (default: None)
        
    Returns:
        dict: Experiment results containing fitted parameters and model comparison
    """
    # Parse parameters
    n_points = kwargs.get('n_points', 50)
    noise_sigma = kwargs.get('noise_sigma', 0.05)
    a_true = kwargs.get('a_true', 2.0)
    beta_true = kwargs.get('beta_true', 1.5)
    output_dir = kwargs.get('output_dir', None)
    
    # Set parameter-dependent random seed for reproducible but parameter-specific results
    seed = _generate_parameter_seed(n_points, noise_sigma, a_true, beta_true)
    np.random.seed(seed)
    
    # Generate synthetic data
    data = generate_synthetic_data(n_points, noise_sigma, a_true, beta_true)
    
    # Fit models
    models = fit_all_models(data['N_values'], data['DoF_values'], data['psi_values'])
    
    # Create results dictionary
    results = {
        'parameters': {
            'n_points': n_points,
            'noise_sigma': noise_sigma,
            'a_true': a_true,
            'beta_true': beta_true
        },
        'fitted_params': {
            'a_hat': models['exponential']['params'][0],
            'a_hat_ci': models['exponential'].get('ci', [None, None]),
            'beta_hat': models['psi_beta']['params'][0],
            'beta_hat_ci': models['psi_beta'].get('ci', [None, None])
        },
        'model_comparison': {
            'exponential': {
                'aic': models['exponential']['aic'],
                'bic': models['exponential']['bic'],
                'cv_score': models['exponential']['cv_score'],
                'cv_std': models['exponential']['cv_std']
            },
            'polynomial': {
                'aic': models['polynomial']['aic'],
                'bic': models['polynomial']['bic'],
                'cv_score': models['polynomial']['cv_score'],
                'cv_std': models['polynomial']['cv_std']
            },
            'power_law': {
                'aic': models['power_law']['aic'],
                'bic': models['power_law']['bic'],
                'cv_score': models['power_law']['cv_score'],
                'cv_std': models['power_law']['cv_std']
            }
        },
        'synthetic_data': {
            'N_values': data['N_values'].tolist(),
            'DoF_values': data['DoF_values'].tolist(),
            'psi_values': data['psi_values'].tolist()
        }
    }
    
    # Save outputs if directory specified
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save demo plots
        fig = run_demo(n_points=n_points, noise_sigma=noise_sigma, 
                      a_true=a_true, beta_true=beta_true)
        fit_path = os.path.join(output_dir, 'bandgaps_fit.png')
        fig.savefig(fit_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Create residuals plot
        fig_res = create_residuals_plot(data, models)
        res_path = os.path.join(output_dir, 'bandgaps_residuals.png')
        fig_res.savefig(res_path, dpi=150, bbox_inches='tight')
        plt.close(fig_res)
        
        # Save results
        results_path = os.path.join(output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        results['files'] = {
            'fit_plot': fit_path,
            'residuals_plot': res_path,
            'results': results_path
        }
    
    return results


def generate_synthetic_data(n_points: int, noise_sigma: float, 
                          a_true: float, beta_true: float) -> Dict[str, np.ndarray]:
    """Generate synthetic bandgaps data."""
    # Sample N uniformly in [0.7, 1.0]
    N_values = np.linspace(0.7, 1.0, n_points)
    
    # True DoF law: DoF(N) = exp(-a*(1-N)) + noise
    DoF_true = np.exp(-a_true * (1 - N_values))
    noise = np.random.normal(0, noise_sigma, n_points)
    DoF_values = DoF_true + noise
    
    # Ensure DoF values are positive
    DoF_values = np.maximum(DoF_values, 0.01)
    
    # Mapping: psi = DoF**beta
    psi_values = DoF_values ** beta_true
    
    return {
        'N_values': N_values,
        'DoF_values': DoF_values,
        'psi_values': psi_values
    }


def exponential_model(N: np.ndarray, a: float) -> np.ndarray:
    """Exponential model: DoF = exp(-a*(1-N))"""
    return np.exp(-a * (1 - N))


def polynomial_model(N: np.ndarray, theta0: float, theta1: float, theta2: float) -> np.ndarray:
    """Polynomial model: DoF = θ0 + θ1*(1-N) + θ2*(1-N)^2"""
    x = 1 - N
    return theta0 + theta1 * x + theta2 * x**2


def power_law_model(N: np.ndarray, c: float, p: float) -> np.ndarray:
    """Power law model: DoF = c * (1-N)^p"""
    # Add small epsilon to avoid division by zero
    x = 1 - N + 1e-10
    return c * (x ** p)


def psi_model(DoF: np.ndarray, beta: float) -> np.ndarray:
    """Psi model: psi = DoF^beta"""
    return DoF ** beta


def fit_all_models(N_values: np.ndarray, DoF_values: np.ndarray, 
                  psi_values: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Fit all models and compute model selection criteria."""
    n = len(N_values)
    models = {}
    
    # Fit exponential model
    try:
        popt_exp, pcov_exp = curve_fit(exponential_model, N_values, DoF_values, p0=[2.0])
        pred_exp = exponential_model(N_values, *popt_exp)
        rss_exp = np.sum((DoF_values - pred_exp)**2)
        k_exp = 1  # number of parameters
        
        models['exponential'] = {
            'params': popt_exp,
            'covariance': pcov_exp,
            'rss': rss_exp,
            'aic': 2*k_exp + n*np.log(rss_exp/n),
            'bic': k_exp*np.log(n) + n*np.log(rss_exp/n),
            'predictions': pred_exp
        }
        
        # Compute confidence intervals (approximate)
        if pcov_exp.shape == (1, 1):
            std_err = np.sqrt(np.diag(pcov_exp))[0]
            models['exponential']['ci'] = [popt_exp[0] - 1.96*std_err, popt_exp[0] + 1.96*std_err]
    
    except Exception as e:
        models['exponential'] = {'params': [np.nan], 'aic': np.inf, 'bic': np.inf, 'predictions': np.nan*np.ones_like(DoF_values)}
    
    # Fit polynomial model
    try:
        popt_poly, pcov_poly = curve_fit(polynomial_model, N_values, DoF_values, p0=[1.0, 1.0, 1.0])
        pred_poly = polynomial_model(N_values, *popt_poly)
        rss_poly = np.sum((DoF_values - pred_poly)**2)
        k_poly = 3  # number of parameters
        
        models['polynomial'] = {
            'params': popt_poly,
            'covariance': pcov_poly,
            'rss': rss_poly,
            'aic': 2*k_poly + n*np.log(rss_poly/n),
            'bic': k_poly*np.log(n) + n*np.log(rss_poly/n),
            'predictions': pred_poly
        }
    
    except Exception as e:
        models['polynomial'] = {'params': [np.nan]*3, 'aic': np.inf, 'bic': np.inf, 'predictions': np.nan*np.ones_like(DoF_values)}
    
    # Fit power law model
    try:
        popt_power, pcov_power = curve_fit(power_law_model, N_values, DoF_values, p0=[1.0, 1.0])
        pred_power = power_law_model(N_values, *popt_power)
        rss_power = np.sum((DoF_values - pred_power)**2)
        k_power = 2  # number of parameters
        
        models['power_law'] = {
            'params': popt_power,
            'covariance': pcov_power,
            'rss': rss_power,
            'aic': 2*k_power + n*np.log(rss_power/n),
            'bic': k_power*np.log(n) + n*np.log(rss_power/n),
            'predictions': pred_power
        }
    
    except Exception as e:
        models['power_law'] = {'params': [np.nan]*2, 'aic': np.inf, 'bic': np.inf, 'predictions': np.nan*np.ones_like(DoF_values)}
    
    # Fit psi model
    try:
        popt_psi, pcov_psi = curve_fit(psi_model, DoF_values, psi_values, p0=[1.5])
        models['psi_beta'] = {
            'params': popt_psi,
            'covariance': pcov_psi
        }
        
        # Compute confidence intervals (approximate)
        if pcov_psi.shape == (1, 1):
            std_err = np.sqrt(np.diag(pcov_psi))[0]
            models['psi_beta']['ci'] = [popt_psi[0] - 1.96*std_err, popt_psi[0] + 1.96*std_err]
    
    except Exception as e:
        models['psi_beta'] = {'params': [np.nan], 'ci': [np.nan, np.nan]}
    
    # Compute cross-validation scores
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, model_func in [('exponential', exponential_model), 
                                   ('polynomial', polynomial_model), 
                                   ('power_law', power_law_model)]:
        cv_scores = []
        
        if model_name in models and not np.any(np.isnan(models[model_name]['params'])):
            for train_idx, val_idx in kf.split(N_values):
                N_train, N_val = N_values[train_idx], N_values[val_idx]
                DoF_train, DoF_val = DoF_values[train_idx], DoF_values[val_idx]
                
                try:
                    # Get initial guess based on full fit
                    p0 = models[model_name]['params']
                    popt, _ = curve_fit(model_func, N_train, DoF_train, p0=p0)
                    pred_val = model_func(N_val, *popt)
                    mse = mean_squared_error(DoF_val, pred_val)
                    cv_scores.append(mse)
                except:
                    cv_scores.append(np.inf)
        
        if cv_scores:
            models[model_name]['cv_score'] = np.mean(cv_scores)
            models[model_name]['cv_std'] = np.std(cv_scores)
        else:
            models[model_name]['cv_score'] = np.inf
            models[model_name]['cv_std'] = np.inf
    
    return models


def create_residuals_plot(data: Dict[str, np.ndarray], 
                         models: Dict[str, Dict[str, Any]]) -> plt.Figure:
    """Create detailed residuals analysis plot."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot residuals vs N
    for i, (name, model) in enumerate([('Exponential', 'exponential'), 
                                      ('Polynomial', 'polynomial'), 
                                      ('Power Law', 'power_law')]):
        if 'predictions' in models[model] and not np.all(np.isnan(models[model]['predictions'])):
            residuals = data['DoF_values'] - models[model]['predictions']
            ax1.scatter(data['N_values'], residuals, alpha=0.7, label=name, s=20)
    
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax1.set_xlabel('N')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs N')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot residuals vs fitted values
    for i, (name, model) in enumerate([('Exponential', 'exponential'), 
                                      ('Polynomial', 'polynomial'), 
                                      ('Power Law', 'power_law')]):
        if 'predictions' in models[model] and not np.all(np.isnan(models[model]['predictions'])):
            residuals = data['DoF_values'] - models[model]['predictions']
            ax2.scatter(models[model]['predictions'], residuals, alpha=0.7, label=name, s=20)
    
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Fitted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Fitted')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Q-Q plot for exponential model
    if 'exponential' in models and 'predictions' in models['exponential']:
        residuals = data['DoF_values'] - models['exponential']['predictions']
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot: Exponential Model')
        ax3.grid(True, alpha=0.3)
    
    # Model comparison bar plot
    if all(model in models for model in ['exponential', 'polynomial', 'power_law']):
        model_names = ['Exponential', 'Polynomial', 'Power Law']
        aic_values = [models['exponential']['aic'], models['polynomial']['aic'], models['power_law']['aic']]
        
        x_pos = np.arange(len(model_names))
        ax4.bar(x_pos, aic_values, alpha=0.7)
        ax4.set_xlabel('Model')
        ax4.set_ylabel('AIC')
        ax4.set_title('Model Comparison (AIC)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(model_names, rotation=45)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
