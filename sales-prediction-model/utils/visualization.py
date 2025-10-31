"""
Visualization utilities for sales prediction models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_time_series(data, title="Time Series Data", figsize=(15, 6)):
    """
    Plot basic time series data.
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(data.index, data.values, linewidth=2, alpha=0.8)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_seasonal_decomposition(decomposition, figsize=(15, 12)):
    """
    Plot seasonal decomposition components.
    
    Parameters:
    -----------
    decomposition : dict
        Dictionary containing decomposition components
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    
    # Original data
    decomposition['observed'].plot(ax=axes[0], title='Original', color='blue')
    axes[0].grid(True, alpha=0.3)
    
    # Trend
    decomposition['trend'].plot(ax=axes[1], title='Trend', color='green')
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal
    decomposition['seasonal'].plot(ax=axes[2], title='Seasonal', color='orange')
    axes[2].grid(True, alpha=0.3)
    
    # Residual
    decomposition['residual'].plot(ax=axes[3], title='Residual', color='red')
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle('Seasonal Decomposition', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_acf_pacf(data, lags=40, figsize=(15, 6)):
    """
    Plot ACF and PACF for time series analysis.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    lags : int
        Number of lags to plot
    figsize : tuple
        Figure size
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ACF plot
    plot_acf(data, lags=lags, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF)', fontsize=14, fontweight='bold')
    
    # PACF plot
    plot_pacf(data, lags=lags, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_stationarity_tests(data, title="Stationarity Analysis"):
    """
    Plot rolling statistics for stationarity analysis.
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    title : str
        Plot title
    """
    # Calculate rolling statistics
    rolling_mean = data.rolling(window=12).mean()
    rolling_std = data.rolling(window=12).std()
    
    plt.figure(figsize=(15, 8))
    
    # Original data
    plt.plot(data.index, data.values, label='Original', alpha=0.7, linewidth=2)
    
    # Rolling statistics
    plt.plot(rolling_mean.index, rolling_mean.values, label='Rolling Mean', color='red', linewidth=2)
    plt.plot(rolling_std.index, rolling_std.values, label='Rolling Std', color='green', linewidth=2)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_differencing(original, diff1, diff2=None, figsize=(15, 10)):
    """
    Plot original and differenced series.
    
    Parameters:
    -----------
    original : pd.Series
        Original time series
    diff1 : pd.Series
        First differenced series
    diff2 : pd.Series, optional
        Second differenced series
    figsize : tuple
        Figure size
    """
    n_plots = 3 if diff2 is not None else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    
    # Original series
    axes[0].plot(original.index, original.values, color='blue', linewidth=2)
    axes[0].set_title('Original Series', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # First difference
    axes[1].plot(diff1.index, diff1.values, color='green', linewidth=2)
    axes[1].set_title('First Difference', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Second difference (if provided)
    if diff2 is not None:
        axes[2].plot(diff2.index, diff2.values, color='red', linewidth=2)
        axes[2].set_title('Second Difference', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_forecast_interactive(train_data, test_data, forecast, 
                            confidence_intervals=None, title="Interactive Forecast"):
    """
    Create interactive forecast plot using Plotly.
    
    Parameters:
    -----------
    train_data : pd.Series
        Training data
    test_data : pd.Series
        Test data
    forecast : array-like
        Forecast values
    confidence_intervals : tuple, optional
        (lower_bound, upper_bound) for confidence intervals
    title : str
        Plot title
    """
    fig = go.Figure()
    
    # Training data
    fig.add_trace(go.Scatter(
        x=train_data.index,
        y=train_data.values,
        mode='lines',
        name='Training Data',
        line=dict(color='blue', width=2)
    ))
    
    # Test data
    fig.add_trace(go.Scatter(
        x=test_data.index,
        y=test_data.values,
        mode='lines+markers',
        name='Actual',
        line=dict(color='green', width=2),
        marker=dict(size=6)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=test_data.index,
        y=forecast,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=6, symbol='square')
    ))
    
    # Confidence intervals
    if confidence_intervals is not None:
        lower_bound, upper_bound = confidence_intervals
        
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            name='95% Confidence Interval',
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title='Date',
        yaxis_title='Sales',
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=500
    )
    
    fig.show()

def plot_model_diagnostics(fitted_model, figsize=(15, 12)):
    """
    Plot comprehensive model diagnostics.
    
    Parameters:
    -----------
    fitted_model : statsmodels fitted model
        Fitted ARIMA/SARIMA model
    figsize : tuple
        Figure size
    """
    # Get residuals
    residuals = fitted_model.resid
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Residuals plot
    axes[0, 0].plot(residuals, alpha=0.7)
    axes[0, 0].set_title('Standardized Residuals', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Histogram of Residuals', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Normal Q-Q Plot', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # ACF of residuals
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, lags=20, ax=axes[1, 1])
    axes[1, 1].set_title('ACF of Residuals', fontweight='bold')
    
    plt.suptitle('Model Diagnostics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_parameter_grid_search(results_df, figsize=(12, 8)):
    """
    Plot grid search results for ARIMA parameters.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with columns: p, d, q, AIC, BIC
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # AIC heatmap
    aic_pivot = results_df.pivot_table(values='AIC', index='p', columns='q', aggfunc='min')
    sns.heatmap(aic_pivot, annot=True, fmt='.0f', ax=axes[0], cmap='viridis')
    axes[0].set_title('AIC Grid Search Results', fontweight='bold')
    
    # BIC heatmap
    bic_pivot = results_df.pivot_table(values='BIC', index='p', columns='q', aggfunc='min')
    sns.heatmap(bic_pivot, annot=True, fmt='.0f', ax=axes[1], cmap='viridis')
    axes[1].set_title('BIC Grid Search Results', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_seasonal_patterns(data, freq='M', figsize=(15, 10)):
    """
    Plot seasonal patterns in the data.
    
    Parameters:
    -----------
    data : pd.Series
        Time series data with datetime index
    freq : str
        Frequency for grouping ('M' for month, 'Q' for quarter)
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    if freq == 'M':
        # Monthly patterns
        monthly_data = data.groupby(data.index.month).mean()
        monthly_data.plot(kind='bar', ax=axes[0, 0], color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Average Sales by Month', fontweight='bold')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].tick_params(axis='x', rotation=0)
        
        # Box plot by month
        df_monthly = pd.DataFrame({'Month': data.index.month, 'Sales': data.values})
        sns.boxplot(data=df_monthly, x='Month', y='Sales', ax=axes[0, 1])
        axes[0, 1].set_title('Sales Distribution by Month', fontweight='bold')
        
    # Quarterly patterns
    quarterly_data = data.groupby(data.index.quarter).mean()
    quarterly_data.plot(kind='bar', ax=axes[1, 0], color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('Average Sales by Quarter', fontweight='bold')
    axes[1, 0].set_xlabel('Quarter')
    axes[1, 0].tick_params(axis='x', rotation=0)
    
    # Year-over-year comparison
    df_yearly = pd.DataFrame({'Year': data.index.year, 'Sales': data.values})
    yearly_data = df_yearly.groupby('Year')['Sales'].mean()
    yearly_data.plot(kind='line', ax=axes[1, 1], marker='o', linewidth=2)
    axes[1, 1].set_title('Average Sales by Year', fontweight='bold')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_forecast_dashboard(train_data, test_data, forecasts_dict):
    """
    Create a comprehensive forecast dashboard.
    
    Parameters:
    -----------
    train_data : pd.Series
        Training data
    test_data : pd.Series
        Test data
    forecasts_dict : dict
        Dictionary with model names as keys and forecasts as values
    """
    # Create subplots
    n_models = len(forecasts_dict)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Forecast Comparison', 'Model Performance', 'Residual Analysis', 'Error Distribution'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Color palette
    colors = px.colors.qualitative.Set1[:n_models]
    
    # Plot 1: Forecast comparison
    # Training data
    fig.add_trace(
        go.Scatter(x=train_data.index, y=train_data.values, 
                  name='Training', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Actual test data
    fig.add_trace(
        go.Scatter(x=test_data.index, y=test_data.values,
                  name='Actual', line=dict(color='black', width=3)),
        row=1, col=1
    )
    
    # Forecasts
    for i, (model_name, forecast) in enumerate(forecasts_dict.items()):
        fig.add_trace(
            go.Scatter(x=test_data.index, y=forecast,
                      name=f'{model_name} Forecast', 
                      line=dict(color=colors[i], width=2, dash='dash')),
            row=1, col=1
        )
    
    # Calculate and plot metrics
    metrics_data = []
    for model_name, forecast in forecasts_dict.items():
        from utils.model_evaluation import calculate_forecast_metrics
        metrics = calculate_forecast_metrics(test_data.values, forecast)
        metrics_data.append({
            'Model': model_name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MAPE': metrics['MAPE']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Plot 2: Model performance
    for i, metric in enumerate(['RMSE', 'MAE', 'MAPE']):
        fig.add_trace(
            go.Bar(x=metrics_df['Model'], y=metrics_df[metric], 
                  name=metric, marker_color=colors[i]),
            row=1, col=2
        )
    
    fig.update_layout(height=800, showlegend=True, 
                     title_text="Sales Forecast Dashboard")
    
    fig.show()