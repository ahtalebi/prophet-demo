# Prophet Time Series Forecasting Demo
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Step 1: Create sample data (or load your own CSV)
def create_sample_data():
    """Create sample time series data with trend and seasonality"""
    
    # Generate dates for 2 years
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create synthetic data with trend + seasonality + noise
    trend = np.linspace(100, 200, len(dates))  # Linear upward trend
    seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)  # Yearly seasonality
    weekly = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Weekly seasonality
    noise = np.random.normal(0, 5, len(dates))  # Random noise
    
    values = trend + seasonal + weekly + noise
    
    # Prophet requires columns named 'ds' (datestamp) and 'y' (value)
    df = pd.DataFrame({
        'ds': dates,
        'y': values
    })
    
    return df

# Step 2: Load or create your data
print("Creating sample data...")
df = create_sample_data()

# Display first few rows
print("Sample data:")
print(df.head())
print(f"Data shape: {df.shape}")

# Step 3: Initialize and fit the Prophet model
print("\nFitting Prophet model...")
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,  # Not needed for daily data
    seasonality_mode='additive'  # or 'multiplicative'
)

# Fit the model
model.fit(df)

# Step 4: Create future dataframe for predictions
print("Creating future predictions...")
# Predict 90 days into the future
future = model.make_future_dataframe(periods=90)
print(f"Future dataframe shape: {future.shape}")

# Step 5: Make predictions
forecast = model.predict(future)

# Display forecast columns
print("\nForecast columns:")
print(forecast.columns.tolist())

# Step 6: Plot the results
print("\nCreating plots...")

# Main forecast plot
fig1 = model.plot(forecast)
plt.title('Prophet Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Components plot (trend, seasonality breakdown)
fig2 = model.plot_components(forecast)
plt.show()

# Step 7: Print some key metrics
print("\nForecast Summary:")
print("Last actual values:")
print(df.tail(3)[['ds', 'y']])

print("\nFirst few predictions:")
future_predictions = forecast[forecast['ds'] > df['ds'].max()]
print(future_predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

# Step 8: Save results (optional)
# forecast.to_csv('prophet_forecast.csv', index=False)
# print("Forecast saved to 'prophet_forecast.csv'")

print("\n=== DEMO COMPLETE ===")
print("Key takeaways:")
print("- 'yhat' is the forecasted value")
print("- 'yhat_lower' and 'yhat_upper' are confidence intervals")
print("- Prophet automatically detected and modeled the seasonality")
print("- The components plot shows trend and seasonal decomposition")
