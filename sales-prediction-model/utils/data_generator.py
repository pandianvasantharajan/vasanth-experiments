"""
Sample data generator for sales prediction models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_monthly_sales_data():
    """Generate monthly sales data for ARIMA modeling."""
    
    # Create date range (5 years of monthly data)
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # Month start
    
    n_months = len(dates)
    
    # Generate base trend
    trend = np.linspace(50000, 80000, n_months)
    
    # Add some noise and cyclical patterns
    noise = np.random.normal(0, 3000, n_months)
    cyclical = 5000 * np.sin(np.linspace(0, 4*np.pi, n_months))
    
    # Marketing spend effect
    marketing_spend = np.random.uniform(5000, 15000, n_months)
    marketing_effect = marketing_spend * 0.3
    
    # Calculate sales
    sales = trend + cyclical + marketing_effect + noise
    sales = np.maximum(sales, 30000)  # Ensure minimum sales
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Sales': sales.round(2),
        'Marketing_Spend': marketing_spend.round(2),
        'Seasonality_Factor': (cyclical / 5000).round(3)
    })
    
    return df

def generate_seasonal_sales_data():
    """Generate seasonal sales data for SARIMA modeling."""
    
    # Create date range (6 years of monthly data)
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    n_months = len(dates)
    
    # Generate base trend
    trend = np.linspace(40000, 90000, n_months)
    
    # Strong seasonal component (Christmas effect)
    months = np.array([d.month for d in dates])
    seasonal = np.where(months == 12, 25000,  # December peak
                       np.where(months == 11, 15000,  # November high
                               np.where(months == 1, -10000,  # January drop
                                       np.where(months == 2, -8000,  # February low
                                               np.where(np.isin(months, [6, 7, 8]), 8000,  # Summer boost
                                                       0)))))
    
    # Add quarterly patterns
    quarters = ((months - 1) // 3) + 1
    quarterly_effect = np.where(quarters == 4, 8000,  # Q4 boost
                               np.where(quarters == 1, -5000,  # Q1 slow
                                       0))
    
    # Holiday effects
    holiday_months = [11, 12, 5, 7]  # Black Friday, Christmas, Memorial Day, July 4th
    holiday_mask = np.isin(months, holiday_months)
    holiday_effect = np.where(holiday_mask, 
                             np.random.uniform(3000, 8000, len(months)), 
                             0)
    
    # Product categories (simulate mix effect)
    categories = ['Electronics', 'Clothing', 'Home', 'Sports']
    product_category = np.random.choice(categories, n_months)
    
    # Category-specific seasonal patterns
    category_effect = np.where(product_category == 'Electronics', seasonal * 1.2,
                              np.where(product_category == 'Clothing', seasonal * 0.8,
                                      np.where(product_category == 'Home', seasonal * 0.6,
                                              seasonal * 1.0)))
    
    # Add noise
    noise = np.random.normal(0, 4000, n_months)
    
    # Calculate total sales
    sales = trend + seasonal + quarterly_effect + category_effect + noise
    sales = np.maximum(sales, 25000)  # Ensure minimum sales
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Sales': sales.round(2),
        'Product_Category': product_category,
        'Holiday_Effect': holiday_effect.round(2),
        'Seasonal_Index': (seasonal / 25000).round(3),
        'Quarter': quarters
    })
    
    return df

if __name__ == "__main__":
    # Generate datasets
    print("Generating monthly sales data for ARIMA...")
    monthly_data = generate_monthly_sales_data()
    monthly_data.to_csv('../data/monthly_sales.csv', index=False)
    print(f"✓ Generated {len(monthly_data)} records in monthly_sales.csv")
    
    print("Generating seasonal sales data for SARIMA...")
    seasonal_data = generate_seasonal_sales_data()
    seasonal_data.to_csv('../data/seasonal_sales.csv', index=False)
    print(f"✓ Generated {len(seasonal_data)} records in seasonal_sales.csv")
    
    print("\nDataset Overview:")
    print("================")
    print(f"Monthly Sales Data: {monthly_data['Date'].min()} to {monthly_data['Date'].max()}")
    print(f"Seasonal Sales Data: {seasonal_data['Date'].min()} to {seasonal_data['Date'].max()}")
    print(f"Monthly Sales Range: ${monthly_data['Sales'].min():,.0f} - ${monthly_data['Sales'].max():,.0f}")
    print(f"Seasonal Sales Range: ${seasonal_data['Sales'].min():,.0f} - ${seasonal_data['Sales'].max():,.0f}")