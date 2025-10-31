"""
Model evaluation utilities for time series forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def calculate_forecast_metrics(actual, predicted):
    """
    Calculate comprehensive forecast evaluation metrics.
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    
    Returns:
    --------
    dict
        Dictionary containing various metrics
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Remove any NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        return {metric: np.nan for metric in ['rmse', 'mae', 'mape', 'smape', 'mase', 'r2']}
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Symmetric Mean Absolute Percentage Error
    smape = np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))) * 100
    
    # Mean Absolute Scaled Error (requires naive forecast for scaling)
    naive_forecast = np.roll(actual, 1)[1:]  # Previous period as forecast
    actual_subset = actual[1:]
    mae_naive = np.mean(np.abs(actual_subset - naive_forecast))
    mase = mae / mae_naive if mae_naive != 0 else np.inf
    
    # R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'SMAPE': smape,
        'MASE': mase,
        'R2': r2
    }

def plot_forecast_results(actual, predicted, train_data=None, title="Forecast Results"):
    """
    Plot actual vs predicted values with training data.
    
    Parameters:
    -----------
    actual : pd.Series
        Actual test values
    predicted : pd.Series or array-like
        Predicted values
    train_data : pd.Series, optional
        Training data to show full context
    title : str
        Title for the plot
    """
    plt.figure(figsize=(15, 8))
    
    # Plot training data if provided
    if train_data is not None:
        plt.plot(train_data.index, train_data.values, 
                label='Training Data', color='blue', alpha=0.7)
    
    # Plot actual and predicted
    plt.plot(actual.index, actual.values, 
            label='Actual', color='green', marker='o', linewidth=2)
    plt.plot(actual.index, predicted, 
            label='Predicted', color='red', marker='s', linewidth=2)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_residuals(actual, predicted, title="Residual Analysis"):
    """
    Plot residual analysis for forecast evaluation.
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    title : str
        Title for the plot
    """
    residuals = np.array(actual) - np.array(predicted)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Residuals vs Fitted
    axes[0, 0].scatter(predicted, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Residuals')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals over time
    axes[1, 1].plot(range(len(residuals)), residuals, marker='o', alpha=0.6)
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Time Index')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def perform_ljung_box_test(residuals, lags=10):
    """
    Perform Ljung-Box test for residual autocorrelation.
    
    Parameters:
    -----------
    residuals : array-like
        Model residuals
    lags : int
        Number of lags to test
    
    Returns:
    --------
    dict
        Test results
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    result = acorr_ljungbox(residuals, lags=lags, return_df=True)
    
    return {
        'test_statistic': result['lb_stat'].iloc[-1],
        'p_value': result['lb_pvalue'].iloc[-1],
        'is_white_noise': result['lb_pvalue'].iloc[-1] > 0.05,
        'full_results': result
    }

def calculate_prediction_intervals(forecast, std_errors, confidence_level=0.95):
    """
    Calculate prediction intervals for forecasts.
    
    Parameters:
    -----------
    forecast : array-like
        Point forecasts
    std_errors : array-like
        Standard errors of forecasts
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% confidence)
    
    Returns:
    --------
    tuple
        (lower_bound, upper_bound)
    """
    from scipy import stats
    
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha/2)
    
    lower_bound = forecast - z_score * std_errors
    upper_bound = forecast + z_score * std_errors
    
    return lower_bound, upper_bound

def cross_validate_time_series(data, model_func, n_splits=5, test_size=12):
    """
    Perform time series cross-validation.
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    model_func : callable
        Function that takes train data and returns fitted model
    n_splits : int
        Number of CV splits
    test_size : int
        Size of test set for each split
    
    Returns:
    --------
    list
        List of metric dictionaries for each split
    """
    results = []
    min_train_size = len(data) - n_splits * test_size
    
    for i in range(n_splits):
        # Define split indices
        train_end = min_train_size + i * test_size
        test_start = train_end
        test_end = test_start + test_size
        
        if test_end > len(data):
            break
        
        # Split data
        train_data = data.iloc[:train_end]
        test_data = data.iloc[test_start:test_end]
        
        try:
            # Fit model and forecast
            model = model_func(train_data)
            forecast = model.forecast(steps=test_size)
            
            # Calculate metrics
            metrics = calculate_forecast_metrics(test_data.values, forecast)
            metrics['split'] = i + 1
            results.append(metrics)
            
        except Exception as e:
            print(f"Error in split {i+1}: {e}")
            continue
    
    return results

def compare_models(results_dict, metrics=['RMSE', 'MAE', 'MAPE']):
    """
    Compare multiple models based on evaluation metrics.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and metric dictionaries as values
    metrics : list
        List of metrics to compare
    
    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    comparison_data = {}
    
    for model_name, model_results in results_dict.items():
        comparison_data[model_name] = {
            metric: model_results.get(metric, np.nan) for metric in metrics
        }
    
    comparison_df = pd.DataFrame(comparison_data).T
    
    # Add ranking for each metric
    for metric in metrics:
        rank_col = f'{metric}_Rank'
        comparison_df[rank_col] = comparison_df[metric].rank(ascending=True)
    
    return comparison_df

def plot_model_comparison(comparison_df, metrics=['RMSE', 'MAE', 'MAPE']):
    """
    Plot model comparison visualization.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Model comparison DataFrame
    metrics : list
        List of metrics to plot
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        comparison_df[metric].plot(kind='bar', ax=axes[i], color='skyblue', alpha=0.7)
        axes[i].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, v in enumerate(comparison_df[metric]):
            axes[i].text(j, v + v*0.01, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def generate_forecast_report(actual, predicted, model_name="Model"):
    """
    Generate a comprehensive forecast evaluation report.
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    model_name : str
        Name of the model
    
    Returns:
    --------
    dict
        Comprehensive report
    """
    metrics = calculate_forecast_metrics(actual, predicted)
    residuals = np.array(actual) - np.array(predicted)
    ljung_box = perform_ljung_box_test(residuals)
    
    report = {
        'model_name': model_name,
        'forecast_metrics': metrics,
        'residual_analysis': {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'ljung_box_test': ljung_box
        },
        'interpretation': {
            'accuracy': 'Good' if metrics['MAPE'] < 10 else 'Moderate' if metrics['MAPE'] < 20 else 'Poor',
            'bias': 'Unbiased' if abs(np.mean(residuals)) < np.std(residuals) * 0.1 else 'Biased',
            'residuals_white_noise': ljung_box['is_white_noise']
        }
    }
    
    return report