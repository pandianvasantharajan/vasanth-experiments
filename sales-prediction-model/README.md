# Sales Prediction Model

A comprehensive time series forecasting project using ARIMA and SARIMA models for sales prediction.

## Project Overview

This project demonstrates sales forecasting using two powerful time series analysis techniques:

1. **ARIMA (AutoRegressive Integrated Moving Average)** - For non-seasonal data
2. **SARIMA (Seasonal ARIMA)** - For data with seasonal patterns

## Features

- 📊 **Comprehensive Data Analysis** - EDA with visualization
- 🔍 **Stationarity Testing** - ADF test, KPSS test
- 📈 **Model Selection** - Auto ARIMA and manual parameter tuning
- 🎯 **Forecasting** - Multi-step ahead predictions
- 📉 **Model Evaluation** - RMSE, MAE, MAPE metrics
- 🎨 **Rich Visualizations** - Interactive plots with Plotly
- 📋 **Detailed Reports** - Model diagnostics and insights

## Project Structure

```
sales-prediction-model/
├── data/
│   ├── monthly_sales.csv          # Sample monthly sales data
│   ├── seasonal_sales.csv         # Sample seasonal sales data
│   └── data_description.md        # Data dictionary
├── notebooks/
│   ├── 01_ARIMA_Sales_Prediction.ipynb    # ARIMA modeling
│   └── 02_SARIMA_Sales_Prediction.ipynb   # SARIMA modeling
├── utils/
│   ├── data_preprocessing.py      # Data preparation utilities
│   ├── model_evaluation.py        # Model evaluation functions
│   └── visualization.py           # Plotting utilities
├── results/
│   ├── arima_forecast.csv         # ARIMA predictions
│   ├── sarima_forecast.csv        # SARIMA predictions
│   └── model_comparison.csv       # Model performance comparison
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd sales-prediction-model
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

## Sample Datasets

### 1. Monthly Sales Data (monthly_sales.csv)
- **Period**: 2019-2023 (60 months)
- **Features**: Date, Sales, Marketing_Spend, Seasonality_Factor
- **Use Case**: ARIMA modeling for trend-based forecasting

### 2. Seasonal Sales Data (seasonal_sales.csv)
- **Period**: 2018-2023 (72 months)
- **Features**: Date, Sales, Product_Category, Holiday_Effect
- **Use Case**: SARIMA modeling for seasonal pattern analysis

## Notebooks

### 1. ARIMA Sales Prediction (01_ARIMA_Sales_Prediction.ipynb)

**Key Sections:**
- Data Loading and EDA
- Stationarity Testing and Transformation
- ACF/PACF Analysis
- Model Parameter Selection
- ARIMA Model Fitting
- Forecasting and Evaluation
- Model Diagnostics

**Learning Outcomes:**
- Understanding ARIMA components (p, d, q)
- Stationarity concepts and transformations
- Model selection techniques
- Forecast interpretation

### 2. SARIMA Sales Prediction (02_SARIMA_Sales_Prediction.ipynb)

**Key Sections:**
- Seasonal Data Analysis
- Seasonal Decomposition
- Seasonal Stationarity Testing
- SARIMA Parameter Selection (P, D, Q, s)
- Model Comparison (ARIMA vs SARIMA)
- Advanced Forecasting Techniques
- Business Insights and Recommendations

**Learning Outcomes:**
- Seasonal pattern identification
- SARIMA parameter tuning
- Seasonal forecasting strategies
- Business application of forecasts

## Key Metrics

Both notebooks include comprehensive evaluation using:

- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **AIC/BIC** (Information Criteria)
- **Ljung-Box Test** (Residual analysis)

## Visualizations

- Time series plots with trend and seasonality
- ACF/PACF correlograms
- Residual diagnostic plots
- Forecast plots with confidence intervals
- Model comparison charts

## Business Applications

1. **Inventory Planning** - Optimize stock levels
2. **Budget Forecasting** - Plan financial resources
3. **Marketing Strategy** - Time promotional campaigns
4. **Resource Allocation** - Staff and capacity planning
5. **Performance Monitoring** - Track against forecasts

## Advanced Features

- **Auto ARIMA** for automated model selection
- **Cross-validation** for robust model evaluation
- **Confidence intervals** for forecast uncertainty
- **Model diagnostics** for assumption validation
- **Interactive plots** for better visualization

## Dependencies

```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
statsmodels>=0.12.0
scikit-learn>=1.0.0
pmdarima>=1.8.0
jupyter>=1.0.0
```

## Usage Examples

### Quick Start with ARIMA
```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Load data
data = pd.read_csv('data/monthly_sales.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Fit ARIMA model
model = ARIMA(data['Sales'], order=(1,1,1))
fitted_model = model.fit()

# Generate forecast
forecast = fitted_model.forecast(steps=12)
```

### Quick Start with SARIMA
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit SARIMA model
model = SARIMAX(data['Sales'], order=(1,1,1), seasonal_order=(1,1,1,12))
fitted_model = model.fit()

# Generate seasonal forecast
forecast = fitted_model.forecast(steps=12)
```

## Getting Started

1. Start with the **ARIMA notebook** if you're new to time series
2. Progress to the **SARIMA notebook** for seasonal data
3. Compare results between both approaches
4. Apply learnings to your own sales data

## Tips for Success

1. **Data Quality** - Ensure clean, consistent time series data
2. **Stationarity** - Always test and achieve stationarity
3. **Model Selection** - Use both statistical tests and visual inspection
4. **Validation** - Use out-of-sample testing for model evaluation
5. **Interpretation** - Focus on business insights, not just accuracy

## Contributing

Feel free to contribute by:
- Adding new datasets
- Improving model implementations
- Enhancing visualizations
- Adding new evaluation metrics

## License

This project is open source and available under the MIT License.

---

**Ready to predict sales?** Open the notebooks and start forecasting! 📈