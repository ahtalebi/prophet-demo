# Prophet Time Series Forecasting Demo

A comprehensive demonstration of Facebook's Prophet library featuring interactive Plotly visualizations and automated GitHub Actions workflows.

![Prophet Forecast](https://img.shields.io/badge/Prophet-Time%20Series-blue)
![Python](https://img.shields.io/badge/Python-3.11+-brightgreen)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-orange)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Automated-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

- **Interactive Visualizations**: Plotly-powered HTML plots with zoom, pan, and hover capabilities
- **Realistic Data**: E-commerce sales simulation with trends, seasonality, and special events
- **Automated Forecasting**: GitHub Actions workflow for scheduled predictions
- **Codespaces Ready**: Optimized for GitHub Codespaces development environment
- **Professional Analysis**: Performance metrics, component decomposition, and confidence intervals
- **No Dependencies Issues**: Browser-based plots (no Chrome/Kaleido requirements)

## 📊 What You'll Get

### Interactive Visualizations
- **Main Forecast Plot**: Historical data, predictions, and confidence intervals
- **Components Breakdown**: Trend, yearly seasonality, weekly patterns, and holiday effects  
- **Performance Analysis**: Model accuracy metrics and residual analysis

### Data Outputs
- **CSV Files**: Complete forecast data and original training dataset
- **Metrics**: MAE, RMSE, MAPE, and R² performance indicators
- **HTML Reports**: Interactive plots viewable in any browser

## 🚀 Quick Start

### Option 1: GitHub Codespaces (Recommended)

1. **Open in Codespaces**: Click the green "Code" button → "Codespaces" → "Create codespace"
2. **Wait for setup**: Environment auto-configures with all dependencies
3. **Run the demo**:
   ```bash
   python prophet_demo.py
   ```
4. **View results**: Open HTML files in `outputs/` folder with browser preview

### Option 2: Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/prophet-demo.git
   cd prophet-demo
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv prophet-env
   source prophet-env/bin/activate  # Linux/macOS
   # prophet-env\Scripts\activate   # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the demo**:
   ```bash
   python prophet_demo.py
   ```

## 📁 Project Structure

```
prophet-demo/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
├── prophet_demo.py             # Main Prophet forecasting script
├── .devcontainer/              # Codespaces configuration
│   └── devcontainer.json
├── .github/workflows/          # GitHub Actions
│   └── prophet-forecast.yml
└── outputs/                    # Generated results
    ├── interactive_forecast.html      # Main forecast plot
    ├── interactive_components.html    # Seasonality breakdown
    ├── performance_analysis.html      # Model metrics
    ├── forecast_results.csv           # Complete forecast data
    ├── original_data.csv             # Training dataset
    └── model_metrics.txt             # Performance summary
```

## 🎯 Data & Model Details

### Synthetic Dataset
- **Time Period**: 3+ years of daily e-commerce sales data
- **Patterns**: Growth trend, yearly seasonality, weekly cycles, holiday effects
- **Special Events**: Black Friday spikes, COVID-19 impact simulation
- **Realism**: Noise, outliers, and business-relevant fluctuations

### Prophet Model Configuration
```python
Prophet(
    growth='linear',                 # Linear growth trend
    yearly_seasonality=True,         # Annual patterns
    weekly_seasonality=True,         # Weekly cycles  
    daily_seasonality=False,         # No daily patterns for daily data
    seasonality_mode='additive',     # Additive seasonality
    changepoint_prior_scale=0.05,    # Trend flexibility
    seasonality_prior_scale=10.0,    # Seasonality strength
    holidays_prior_scale=10.0        # Holiday effect strength
)
```

### Features Added
- US holidays integration
- Custom monthly seasonality
- Confidence interval estimation
- Robust outlier handling

## 🤖 GitHub Actions Automation

### Automated Workflows
- **On Push**: Runs forecast when code changes
- **Weekly Schedule**: Updates predictions every Monday at 9 AM UTC
- **Manual Trigger**: Run on-demand with custom parameters
- **Quality Checks**: Code linting and formatting validation

### Workflow Features
- Dependency caching for faster runs
- Artifact uploads (HTML plots, CSV data, metrics)
- Auto-commit results back to repository
- Comprehensive logging and error handling

### Manual Execution
1. Go to **Actions** tab in your GitHub repository
2. Select **"Prophet Forecasting Demo"**
3. Click **"Run workflow"**
4. Optionally set custom forecast period
5. Download artifacts after completion

## 📈 Viewing Results

### In Codespaces
1. Navigate to `outputs/` folder
2. Right-click any `.html` file
3. Select "Open with Live Preview" or similar option
4. Interactive plots open in browser tab

### Download & Local Viewing
1. Download HTML files from `outputs/` folder
2. Open in your preferred web browser
3. Enjoy full interactivity (zoom, pan, hover)

### GitHub Actions Artifacts
1. Check completed workflow runs
2. Download artifact bundles:
   - `prophet-interactive-plots` (HTML files)
   - `prophet-forecast-data` (CSV files)  
   - `prophet-model-metrics` (performance data)

## 🔧 Customization

### Modify Forecast Period
```python
future = model.make_future_dataframe(periods=365)  # 1 year forecast
```

### Adjust Model Parameters
```python
model = Prophet(
    changepoint_prior_scale=0.1,    # More flexible trend
    seasonality_prior_scale=15.0,   # Stronger seasonality
    interval_width=0.9              # 90% confidence intervals
)
```

### Add Custom Seasonalities
```python
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
```

### Holiday Integration
```python
# Built-in country holidays
model.add_country_holidays(country_name='US')

# Custom holidays
holidays = pd.DataFrame({
    'holiday': 'custom_event',
    'ds': pd.to_datetime(['2024-01-15', '2024-07-15']),
    'lower_window': 0,
    'upper_window': 1,
})
```

## 📊 Performance Metrics

The model automatically calculates:
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Square Error): Penalizes large errors
- **MAPE** (Mean Absolute Percentage Error): Relative error percentage
- **R²** (Coefficient of Determination): Explained variance

## 🛠️ Dependencies

```txt
prophet>=1.1.0          # Facebook's forecasting library
plotly>=5.15.0          # Interactive plotting
pandas>=1.5.0           # Data manipulation
numpy>=1.21.0           # Numerical operations
```

## 🌐 Browser Compatibility

Interactive HTML plots work in:
- ✅ Chrome/Chromium
- ✅ Firefox  
- ✅ Safari
- ✅ Edge
- ✅ GitHub Codespaces browser preview

## 🚨 Troubleshooting

### Common Issues

**Plots not displaying in Codespaces:**
- Try different "Open with" options for HTML files
- Use browser preview extensions
- Download and open locally

**GitHub Actions failing:**
- Check Python version compatibility
- Verify requirements.txt is up to date
- Review workflow logs for specific errors

**Model performance poor:**
- Increase training data period
- Adjust seasonality parameters
- Add relevant holidays or custom seasonalities

### Getting Help

1. Check the **Issues** tab for common problems
2. Review **GitHub Actions** logs for error details
3. Examine `outputs/model_metrics.txt` for performance insights

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Add tests if applicable
5. Commit changes (`git commit -am 'Add new feature'`)
6. Push to branch (`git push origin feature/improvement`)
7. Create a Pull Request

## 📚 Learning Resources

- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [Time Series Forecasting Guide](https://facebook.github.io/prophet/docs/quick_start.html)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## 🎯 Use Cases

This demo is perfect for:
- **Business Forecasting**: Sales, revenue, demand prediction
- **Educational Purposes**: Learning time series analysis
- **Portfolio Projects**: Demonstrating data science skills
- **Production Templates**: Starting point for real forecasting systems

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Facebook's Core Data Science Team** for creating Prophet
- **Plotly Team** for outstanding visualization tools  
- **GitHub** for Codespaces and Actions platforms
- **Open Source Community** for continuous improvements

---

## 🚀 Ready to Forecast?

1. **Click** the green "Use this template" button
2. **Open** in GitHub Codespaces  
3. **Run** `python prophet_demo.py`
4. **Explore** your interactive forecasts!

**Happy Forecasting! 📈✨**