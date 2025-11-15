# Time Series Forecasting (TSF) Experiments

This project contains experiments and implementations for time series forecasting using various statistical and machine learning approaches.

## Overview

Time series forecasting is a critical technique for predicting future values based on previously observed values. This project focuses on classical statistical methods, particularly Autoregressive (AR) models.

## Project Structure

```
tsf-experiments/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── notebooks/                   # Jupyter notebooks
│   └── ar1_model_analysis.ipynb # AR(1) model implementation
├── data/                        # Time series datasets
├── results/                     # Output files and plots
└── models/                      # Saved model files
```

## Features

- **AR(1) Model**: Autoregressive model of order 1
  - Parameter estimation using OLS and MLE
  - Model diagnostics and validation
  - Forecasting and prediction intervals
  - Residual analysis
  - Model evaluation metrics

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/ar1_model_analysis.ipynb
```

### AR(1) Model

The AR(1) (Autoregressive model of order 1) is defined as:

```
Y_t = φ₀ + φ₁ * Y_{t-1} + ε_t
```

Where:
- Y_t: Current value at time t
- Y_{t-1}: Previous value at time t-1
- φ₀: Constant/intercept term
- φ₁: Autoregressive coefficient
- ε_t: White noise error term

## Key Concepts

### Stationarity
For an AR(1) process to be stationary, |φ₁| < 1

### Parameter Interpretation
- φ₁ > 0: Positive autocorrelation (values tend to persist)
- φ₁ < 0: Negative autocorrelation (values tend to oscillate)
- φ₁ ≈ 0: No autocorrelation (random walk)

## Examples

```python
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

# Sample data
Y = [50, 54, 56, 53, 52, 55, 57]

# Fit AR(1) model
model = AutoReg(Y, lags=1)
results = model.fit()

# Make predictions
forecast = results.predict(start=len(Y), end=len(Y)+2)
```

## Model Diagnostics

- **ACF/PACF Plots**: Check autocorrelation structure
- **Residual Analysis**: Verify white noise assumption
- **AIC/BIC**: Model selection criteria
- **Ljung-Box Test**: Test for residual autocorrelation
- **Normality Tests**: Check if residuals are normally distributed

## References

- Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). Time Series Analysis: Forecasting and Control
- Hamilton, J. D. (1994). Time Series Analysis
- Brockwell, P. J., & Davis, R. A. (2016). Introduction to Time Series and Forecasting

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
