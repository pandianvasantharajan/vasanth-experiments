# Air Temperature Forecasting Model

## Project Overview

This project implements a comprehensive time-series forecasting system for predicting air temperature using multiple modeling approaches. The focus is on achieving high accuracy, robustness, interpretability, and reproducibility.

## Project Structure

```
forecasting-model/
├── notebooks/
│   └── air_temperature_forecasting.ipynb    # Main analysis notebook
├── data/
│   └── sample_temperature_data.csv          # Generated sample dataset
├── models/                                  # Saved model artifacts
├── results/                                 # Model outputs and comparisons
├── utils/                                   # Utility functions
├── requirements.txt                         # Python dependencies
└── README.md                               # This file
```

## Features

### Data Analysis
- **Exploratory Data Analysis**: Comprehensive visualization and statistical analysis
- **Time Series Decomposition**: Trend, seasonal, and residual component analysis
- **Feature Engineering**: Lag features, rolling statistics, cyclical encoding

### Models Implemented
- **Baseline Models**: Naive, Seasonal Naive, Moving Average
- **Statistical Models**: ARIMA/SARIMA with automatic parameter selection
- **Machine Learning**: Random Forest, XGBoost with extensive feature engineering
- **Deep Learning**: LSTM neural networks (optional, requires TensorFlow)

### Evaluation Framework
- **Comprehensive Metrics**: RMSE, MAE, MAPE, R², Directional Accuracy
- **Multiple Horizons**: 1-hour, 6-hour, 24-hour, 1-week ahead predictions
- **Visual Analysis**: Prediction plots, residual analysis, feature importance
- **Model Comparison**: Automated ranking and statistical comparison

## Quick Start

### 1. Environment Setup

**Prerequisites:**
- Python 3.9 or higher
- [Poetry](https://python-poetry.org/) for dependency management

**Install Poetry** (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
# Add to PATH: export PATH="$HOME/.local/bin:$PATH"
```

**Set up the project:**
```bash
# Clone or navigate to the project directory
cd forecasting-model

# Install all dependencies (including dev dependencies)
poetry install

# Activate the virtual environment
poetry shell

# Alternative: run commands with poetry run
poetry run jupyter notebook
```

### 2. Run the Analysis

```bash
# Start Jupyter notebook
poetry run jupyter notebook

# Or if you activated the shell:
jupyter notebook

# Open and run the main notebook
# notebooks/air_temperature_forecasting.ipynb
```

### 3. Development Commands

```bash
# Install new dependencies
poetry add package-name

# Install development dependencies
poetry add --group dev package-name

# Update dependencies
poetry update

# Run code formatting
poetry run black .
poetry run isort .

# Run linting
poetry run flake8

# Run tests
poetry run pytest

# Run type checking
poetry run mypy .
```

### 3. View Results

Results are automatically saved to the `results/` directory:
- Model comparison metrics in CSV format
- Visualizations and plots
- Feature importance analysis

## Key Objectives

### 🎯 **Accuracy**
- Multiple model architectures to capture different patterns
- Comprehensive hyperparameter optimization
- Feature engineering for improved performance

### 🔧 **Robustness**
- Cross-validation for time series data
- Multiple evaluation metrics
- Residual analysis and model diagnostics

### 📊 **Interpretability**
- Feature importance rankings
- Model coefficient analysis
- Visual explanation of predictions

### 🔬 **Reproducibility**
- Fixed random seeds across all models
- Comprehensive logging and documentation
- Modular, well-documented code

## Model Performance

The notebook implements and compares multiple approaches:

| Model Type | Advantages | Use Case |
|------------|------------|----------|
| **Baseline** | Simple, fast, interpretable | Quick prototyping, benchmarking |
| **ARIMA** | Captures temporal dependencies | Statistical analysis, trend understanding |
| **Random Forest** | Feature importance, robust | Good balance of accuracy and interpretability |
| **XGBoost** | High accuracy, handles non-linearity | Production forecasting systems |
| **LSTM** | Captures complex patterns | Long-term dependencies, complex data |

## Usage Examples

### Basic Forecasting
```python
# Load the trained model
model = load_model('models/best_model.pkl')

# Make predictions
predictions = model.predict(new_data)

# Evaluate
metrics = calculate_metrics(actual, predictions)
```

### Custom Feature Engineering
```python
# Add custom features
df['custom_feature'] = create_custom_feature(df)

# Retrain models with new features
retrain_models(df, feature_list)
```

## Data Requirements

The system expects hourly temperature data with the following format:

```
datetime,temperature
2019-01-01 00:00:00,5.2
2019-01-01 01:00:00,4.8
...
```

**Required columns:**
- `datetime`: Timestamp in ISO format
- `temperature`: Air temperature in Celsius

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Run tests and ensure reproducibility
5. Submit a pull request

## Dependencies

This project uses [Poetry](https://python-poetry.org/) for dependency management. All dependencies are defined in `pyproject.toml`.

**Core libraries:**
- **pandas, numpy**: Data manipulation and numerical computing
- **matplotlib, seaborn, plotly**: Comprehensive visualization suite
- **scikit-learn**: Machine learning algorithms and metrics
- **statsmodels**: Statistical and time series analysis
- **xgboost**: Gradient boosting framework
- **tensorflow/keras**: Deep learning (optional, use `poetry install -E deep-learning`)

**Development tools:**
- **black, isort**: Code formatting
- **flake8, mypy**: Linting and type checking
- **pytest**: Testing framework
- **jupyter, jupyterlab**: Interactive notebooks

**Optional extras:**
```bash
# Install with deep learning support
poetry install -E deep-learning

# Install all optional dependencies
poetry install -E all
```

See `pyproject.toml` for complete dependency specifications.

## License

This project is open source and available under the MIT License.

## Citation

If you use this forecasting framework in your research, please cite:

```
Air Temperature Forecasting Model
Comprehensive Time Series Analysis Framework
[Your Name/Organization], 2024
```

## Contact

For questions, issues, or contributions, please:
- Open an issue in the repository
- Contact: [your-email@example.com]

---

**Note**: This project includes synthetic temperature data for demonstration. For production use, replace with actual meteorological data from reliable sources.