# Prophet Time Series Forecasting Demo

A simple demonstration of Facebook's Prophet library for time series forecasting and visualization.

![Prophet Forecast](https://img.shields.io/badge/Prophet-Time%20Series-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

This project demonstrates how to use Facebook's Prophet library to:
- Generate synthetic time series data with trends and seasonality
- Fit a Prophet forecasting model
- Create predictions with confidence intervals
- Visualize results with trend and seasonal component decomposition

## Features

- **Automatic seasonality detection**: Yearly and weekly patterns
- **Robust forecasting**: Handles missing data and outliers well
- **Visual components breakdown**: Separate trend and seasonality plots
- **Confidence intervals**: Upper and lower prediction bounds
- **Easy to customize**: Modify parameters for different data types

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/prophet-demo.git
   cd prophet-demo
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv prophet-env
   source prophet-env/bin/activate  # Linux/macOS
   # prophet-env\Scripts\activate   # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Demo

Run the main demo script:
```bash
python prophet_demo.py
```

This will:
1. Generate 2 years of synthetic daily data
2. Fit a Prophet model
3. Forecast 90 days into the future
4. Display forecast and component plots

### Using Your Own Data

To use your own data, modify the data loading section:

```python
# Replace the sample data creation with:
df = pd.read_csv('your_data.csv')
df = df.rename(columns={'date_column': 'ds', 'value_column': 'y'})
df['ds'] = pd.to_datetime(df['ds'])
```

**Data Requirements:**
- CSV file with date and numeric value columns
- Dates should be in a parseable format (YYYY-MM-DD recommended)
- At least several months of historical data for best results

## Output

The script generates:

1. **Forecast Plot**: Shows historical data, predictions, and confidence intervals
2. **Components Plot**: Breaks down the forecast into:
   - Overall trend
   - Yearly seasonality
   - Weekly seasonality
3. **Console Output**: Key statistics and sample predictions

## Customization

### Model Parameters

Adjust the Prophet model for different data characteristics:

```python
model = Prophet(
    growth='linear',              # 'linear' or 'logistic'
    yearly_seasonality=True,      # Enable yearly patterns
    weekly_seasonality=True,      # Enable weekly patterns
    daily_seasonality=False,      # Enable daily patterns
    seasonality_mode='additive', # 'additive' or 'multiplicative'
    changepoint_prior_scale=0.05, # Trend flexibility (0.001-0.5)
    seasonality_prior_scale=10.0   # Seasonality strength (0.01-10+)
)
```

### Forecasting Period

Change the prediction horizon:
```python
# Predict different periods into the future
future = model.make_future_dataframe(periods=30)   # 30 days
future = model.make_future_dataframe(periods=365)  # 1 year
```

## Project Structure

```
prophet-demo/
├── README.md
├── requirements.txt
├── .gitignore
├── prophet_demo.py
├── data/ (optional)
│   └── your_data.csv
└── outputs/ (created when running)
    └── prophet_forecast.csv
```

## Dependencies

- `prophet>=1.1.0` - Facebook's forecasting library
- `pandas>=1.3.0` - Data manipulation
- `matplotlib>=3.3.0` - Basic plotting
- `numpy>=1.21.0` - Numerical computations

## Troubleshooting

### Installation Issues

**On Windows:**
```bash
# If Prophet installation fails, try:
pip install prophet --no-cache-dir
# You may need Visual Studio Build Tools
```

**On macOS:**
```bash
# If you get compilation errors:
xcode-select --install
pip install prophet
```

### Common Issues

1. **Plots not showing**: Add `plt.show()` or run in Jupyter notebook
2. **Date parsing errors**: Ensure date column is properly formatted
3. **Memory issues**: Reduce data size or forecast period for large datasets

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## Resources

- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Prophet GitHub Repository](https://github.com/facebook/prophet)
- [Time Series Analysis with Prophet](https://research.fb.com/blog/2017/02/prophet-forecasting-at-scale/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Facebook's Core Data Science team for creating Prophet
- The open-source community for continuous improvements

---

**Happy Forecasting! 📈**
