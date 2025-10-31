"""
Data preprocessing utilities for sales prediction models.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path, date_column='Date', target_column='Sales'):
    """
    Load and prepare time series data for modeling.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    date_column : str
        Name of the date column
    target_column : str
        Name of the target variable column
    
    Returns:
    --------
    pd.Series
        Time series data with datetime index
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Set date as index
    df.set_index(date_column, inplace=True)
    
    # Sort by date
    df.sort_index(inplace=True)
    
    # Return target series
    return df[target_column]

def check_missing_values(series):
    """
    Check for missing values in time series.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    
    Returns:
    --------
    dict
        Information about missing values
    """
    missing_info = {
        'total_missing': series.isnull().sum(),
        'percent_missing': (series.isnull().sum() / len(series)) * 100,
        'missing_dates': series[series.isnull()].index.tolist()
    }
    
    return missing_info

def handle_missing_values(series, method='interpolate'):
    """
    Handle missing values in time series.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    method : str
        Method to handle missing values ('interpolate', 'forward_fill', 'backward_fill')
    
    Returns:
    --------
    pd.Series
        Time series with missing values handled
    """
    if method == 'interpolate':
        return series.interpolate(method='time')
    elif method == 'forward_fill':
        return series.fillna(method='ffill')
    elif method == 'backward_fill':
        return series.fillna(method='bfill')
    else:
        raise ValueError("Method must be 'interpolate', 'forward_fill', or 'backward_fill'")

def detect_outliers(series, method='iqr', threshold=1.5):
    """
    Detect outliers in time series data.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    method : str
        Method to detect outliers ('iqr', 'zscore')
    threshold : float
        Threshold for outlier detection
    
    Returns:
    --------
    pd.Series
        Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (series < lower_bound) | (series > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = z_scores > threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return outliers

def handle_outliers(series, outliers, method='replace'):
    """
    Handle outliers in time series data.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    outliers : pd.Series
        Boolean series indicating outliers
    method : str
        Method to handle outliers ('replace', 'remove')
    
    Returns:
    --------
    pd.Series
        Time series with outliers handled
    """
    if method == 'replace':
        # Replace outliers with median
        series_cleaned = series.copy()
        series_cleaned[outliers] = series.median()
        return series_cleaned
    
    elif method == 'remove':
        # Remove outliers
        return series[~outliers]
    
    else:
        raise ValueError("Method must be 'replace' or 'remove'")

def create_lag_features(series, lags):
    """
    Create lag features for time series analysis.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    lags : list
        List of lag periods
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with lag features
    """
    df = pd.DataFrame({'original': series})
    
    for lag in lags:
        df[f'lag_{lag}'] = series.shift(lag)
    
    return df

def create_moving_averages(series, windows):
    """
    Create moving average features.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    windows : list
        List of window sizes
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with moving average features
    """
    df = pd.DataFrame({'original': series})
    
    for window in windows:
        df[f'ma_{window}'] = series.rolling(window=window).mean()
        df[f'ma_{window}_std'] = series.rolling(window=window).std()
    
    return df

def split_train_test(series, test_size=0.2, split_method='last'):
    """
    Split time series data into train and test sets.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    test_size : float or int
        Size of test set (fraction or number of observations)
    split_method : str
        Method to split data ('last', 'random')
    
    Returns:
    --------
    tuple
        (train_series, test_series)
    """
    if isinstance(test_size, float):
        test_size = int(len(series) * test_size)
    
    if split_method == 'last':
        train = series[:-test_size]
        test = series[-test_size:]
    elif split_method == 'random':
        # For time series, random split is not recommended
        # but included for completeness
        indices = np.random.choice(len(series), test_size, replace=False)
        test_indices = series.index[indices]
        train_indices = series.index[~series.index.isin(test_indices)]
        train = series[train_indices].sort_index()
        test = series[test_indices].sort_index()
    else:
        raise ValueError("split_method must be 'last' or 'random'")
    
    return train, test

def calculate_seasonal_decomposition(series, model='additive', period=None):
    """
    Perform seasonal decomposition of time series.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    model : str
        Type of decomposition ('additive', 'multiplicative')
    period : int
        Seasonal period (if None, automatically detected)
    
    Returns:
    --------
    dict
        Dictionary containing trend, seasonal, and residual components
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    if period is None:
        # Simple heuristic for period detection
        freq = pd.infer_freq(series.index)
        if freq:
            if 'M' in freq:
                period = 12  # Monthly data
            elif 'D' in freq:
                period = 365  # Daily data
            elif 'Q' in freq:
                period = 4   # Quarterly data
        else:
            period = 12  # Default assumption
    
    decomposition = seasonal_decompose(series, model=model, period=period)
    
    return {
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid,
        'observed': decomposition.observed
    }

def prepare_data_summary(series):
    """
    Generate comprehensive summary of time series data.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    
    Returns:
    --------
    dict
        Summary statistics and information
    """
    summary = {
        'basic_stats': {
            'count': len(series),
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'range': series.max() - series.min()
        },
        'time_info': {
            'start_date': series.index.min(),
            'end_date': series.index.max(),
            'frequency': pd.infer_freq(series.index),
            'total_periods': len(series)
        },
        'data_quality': {
            'missing_values': series.isnull().sum(),
            'missing_percentage': (series.isnull().sum() / len(series)) * 100,
            'duplicated_indices': series.index.duplicated().sum()
        }
    }
    
    return summary