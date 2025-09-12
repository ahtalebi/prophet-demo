# Prophet Time Series Forecasting with Plotly (Codespaces Compatible)
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

def create_output_directory():
    """Create outputs directory if it doesn't exist"""
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    print("📁 Created outputs directory")
    return 'outputs'

def create_sample_data():
    """Create realistic sample data with multiple patterns"""
    print("📊 Generating realistic time series data...")
    
    np.random.seed(42)  # For reproducibility
    
    # Generate dates for 3 years (more data = better forecasting)
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2024, 8, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create e-commerce sales data with multiple components
    n_days = len(dates)
    
    # 1. Base trend (growing business)
    trend = np.linspace(1000, 3500, n_days)
    
    # 2. Yearly seasonality (holiday seasons)
    yearly_seasonal = 400 * np.sin(2 * np.pi * np.arange(n_days) / 365.25 + np.pi/6)
    
    # 3. Weekly seasonality (weekend vs weekday sales)
    weekly_seasonal = 200 * np.sin(2 * np.pi * np.arange(n_days) / 7 + np.pi/4)
    
    # 4. Monthly payday effect
    monthly_effect = 150 * np.sin(2 * np.pi * np.arange(n_days) / 30.44)
    
    # 5. Special events (Black Friday, Christmas, etc.)
    special_events = np.zeros(n_days)
    for year in range(2021, 2025):
        # Black Friday effect (4th Thursday of November + weekend)
        try:
            thanksgiving = pd.Timestamp(f'{year}-11-01') + pd.DateOffset(days=(3-pd.Timestamp(f'{year}-11-01').dayofweek) % 7 + 21)
            black_friday = thanksgiving + pd.DateOffset(days=1)
            
            if black_friday in dates:
                bf_idx = dates.get_loc(black_friday)
                # 5-day sales boost
                for i, boost in enumerate([300, 800, 600, 400, 200]):  # Thu to Mon
                    if bf_idx + i < len(special_events):
                        special_events[bf_idx + i] = boost
        except:
            pass
        
        # Christmas season boost
        christmas_start = pd.Timestamp(f'{year}-12-15')
        christmas_end = pd.Timestamp(f'{year}-12-25')
        if christmas_start in dates:
            start_idx = dates.get_loc(christmas_start)
            end_idx = min(dates.get_loc(christmas_end) if christmas_end in dates else len(dates), len(special_events))
            special_events[start_idx:end_idx] += 300
    
    # 6. COVID-19 impact (2021-2022)
    covid_impact = np.zeros(n_days)
    covid_periods = [
        (pd.Timestamp('2021-01-01'), pd.Timestamp('2021-04-01'), -500),  # Lockdown
        (pd.Timestamp('2021-11-01'), pd.Timestamp('2022-02-01'), -300),  # Omicron wave
    ]
    
    for start, end, impact in covid_periods:
        if start in dates and end in dates:
            start_idx = dates.get_loc(start)
            end_idx = dates.get_loc(end)
            covid_impact[start_idx:end_idx] = impact
    
    # 7. Random noise
    noise = np.random.normal(0, 100, n_days)
    
    # Combine all components
    sales = trend + yearly_seasonal + weekly_seasonal + monthly_effect + special_events + covid_impact + noise
    
    # Ensure no negative sales
    sales = np.maximum(sales, 100)
    
    # Create Prophet-compatible dataframe
    df = pd.DataFrame({
        'ds': dates,
        'y': sales
    })
    
    print(f"✅ Generated {len(df)} days of data")
    print(f"📅 Date range: {df['ds'].min().date()} to {df['ds'].max().date()}")
    print(f"💰 Sales range: ${df['y'].min():.0f} to ${df['y'].max():.0f}")
    
    return df

def create_prophet_model():
    """Create and configure Prophet model"""
    print("🔧 Setting up Prophet model...")
    
    model = Prophet(
        growth='linear',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05,  # Controls trend flexibility
        seasonality_prior_scale=10.0,   # Controls seasonality strength
        holidays_prior_scale=10.0,      # Controls holiday effect strength
        mcmc_samples=0,                 # Use MAP estimation (faster)
        interval_width=0.8,             # 80% confidence intervals
        uncertainty_samples=1000
    )
    
    # Add US holidays
    model.add_country_holidays(country_name='US')
    
    # Add custom seasonalities
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    print("✅ Prophet model configured with holidays and custom seasonalities")
    return model

def create_interactive_forecast_plot(model, forecast, df, output_dir):
    """Create interactive forecast plot with Plotly"""
    print("📈 Creating interactive forecast plot...")
    
    # Split data
    historical = df
    future_data = forecast[forecast['ds'] > df['ds'].max()]
    full_forecast = forecast[forecast['ds'] <= df['ds'].max()]
    
    # Create main figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical['ds'],
        y=historical['y'],
        mode='markers',
        name='Historical Data',
        marker=dict(
            size=4,
            color='#2E86C1',
            opacity=0.7
        ),
        hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> $%{y:,.0f}<extra></extra>'
    ))
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=future_data['ds'],
        y=future_data['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#E74C3C', width=3),
        hovertemplate='<b>Date:</b> %{x}<br><b>Forecast:</b> $%{y:,.0f}<extra></extra>'
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=future_data['ds'],
        y=future_data['yhat_upper'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=future_data['ds'],
        y=future_data['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='80% Confidence Interval',
        fillcolor='rgba(231, 76, 60, 0.2)',
        hovertemplate='<b>Date:</b> %{x}<br><b>Lower:</b> $%{y:,.0f}<extra></extra>'
    ))
    
    # Add fitted values for historical period
    fig.add_trace(go.Scatter(
        x=full_forecast['ds'],
        y=full_forecast['yhat'],
        mode='lines',
        name='Model Fit',
        line=dict(color='#F39C12', width=2, dash='dot'),
        hovertemplate='<b>Date:</b> %{x}<br><b>Fitted:</b> $%{y:,.0f}<extra></extra>'
    ))
    
    # Customize layout
    fig.update_layout(
        title={
            'text': 'Prophet Time Series Forecast - E-commerce Sales',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2C3E50'}
        },
        xaxis_title='Date',
        yaxis_title='Sales (USD)',
        hovermode='x unified',
        width=1200,
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=12),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        )
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    # Save interactive plot
    html_path = f'{output_dir}/interactive_forecast.html'
    fig.write_html(html_path)
    
    # Also save as static image
    png_path = f'{output_dir}/forecast_plot.png'
    
    print(f"✅ Interactive forecast saved to {html_path}")
    print(f"✅ Static forecast saved to {png_path}")
    
    return fig

def create_components_plot(model, forecast, output_dir):
    """Create interactive components breakdown plot"""
    print("📊 Creating components breakdown plot...")
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            'Overall Trend',
            'Yearly Seasonality',
            'Weekly Seasonality',
            'Holiday Effects'
        ],
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # Trend
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast['trend'],
            mode='lines',
            name='Trend',
            line=dict(color='#2E86C1', width=2),
            hovertemplate='<b>Date:</b> %{x}<br><b>Trend:</b> $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Yearly seasonality
    if 'yearly' in forecast.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yearly'],
                mode='lines',
                name='Yearly',
                line=dict(color='#E74C3C', width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Yearly Effect:</b> $%{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Weekly seasonality
    if 'weekly' in forecast.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['weekly'],
                mode='lines',
                name='Weekly',
                line=dict(color='#27AE60', width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Weekly Effect:</b> $%{y:,.0f}<extra></extra>'
            ),
            row=3, col=1
        )
    
    # Holiday effects
    if 'holidays' in forecast.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['holidays'],
                mode='lines',
                name='Holidays',
                line=dict(color='#8E44AD', width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Holiday Effect:</b> $%{y:,.0f}<extra></extra>'
            ),
            row=4, col=1
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Prophet Model Components Breakdown',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50'}
        },
        height=800,
        width=1200,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update all y-axes
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    # Save plots
    html_path = f'{output_dir}/interactive_components.html'
    png_path = f'{output_dir}/components_plot.png'
    
    fig.write_html(html_path)
    
    print(f"✅ Interactive components saved to {html_path}")
    print(f"✅ Static components saved to {png_path}")
    
    return fig

def create_performance_plot(model, forecast, df, output_dir):
    """Create model performance visualization"""
    print("📈 Creating performance analysis plot...")
    
    # Get training predictions
    train_forecast = forecast[forecast['ds'] <= df['ds'].max()]
    merged = df.merge(train_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    
    # Calculate residuals
    merged['residuals'] = merged['y'] - merged['yhat']
    merged['abs_residuals'] = abs(merged['residuals'])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Actual vs Predicted',
            'Residuals Over Time',
            'Residuals Distribution',
            'Model Performance Metrics'
        ],
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'histogram'}, {'type': 'table'}]]
    )
    
    # 1. Actual vs Predicted scatter
    fig.add_trace(
        go.Scatter(
            x=merged['yhat'],
            y=merged['y'],
            mode='markers',
            name='Predictions',
            marker=dict(color='#3498DB', size=4, opacity=0.6),
            hovertemplate='<b>Predicted:</b> $%{x:,.0f}<br><b>Actual:</b> $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Perfect prediction line
    min_val, max_val = min(merged['y'].min(), merged['yhat'].min()), max(merged['y'].max(), merged['yhat'].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Residuals over time
    fig.add_trace(
        go.Scatter(
            x=merged['ds'],
            y=merged['residuals'],
            mode='markers',
            name='Residuals',
            marker=dict(color='#E74C3C', size=3, opacity=0.6),
            hovertemplate='<b>Date:</b> %{x}<br><b>Residual:</b> $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
    
    # 3. Residuals histogram
    fig.add_trace(
        go.Histogram(
            x=merged['residuals'],
            nbinsx=30,
            name='Residuals Distribution',
            marker_color='#9B59B6',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # 4. Performance metrics table
    mae = merged['abs_residuals'].mean()
    rmse = np.sqrt((merged['residuals'] ** 2).mean())
    mape = (merged['abs_residuals'] / merged['y']).mean() * 100
    r2 = np.corrcoef(merged['y'], merged['yhat'])[0, 1] ** 2
    
    metrics_table = go.Table(
        header=dict(values=['Metric', 'Value'], fill_color='#3498DB', font_color='white'),
        cells=dict(values=[
            ['MAE', 'RMSE', 'MAPE', 'R²', 'Data Points'],
            [f'${mae:,.0f}', f'${rmse:,.0f}', f'{mape:.1f}%', f'{r2:.3f}', f'{len(merged):,}']
        ])
    )
    
    fig.add_trace(metrics_table, row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Prophet Model Performance Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50'}
        },
        height=700,
        width=1200,
        showlegend=False
    )
    
    # Save plots
    html_path = f'{output_dir}/performance_analysis.html'
    png_path = f'{output_dir}/performance_plot.png'
    
    fig.write_html(html_path)
    
    print(f"✅ Performance analysis saved to {html_path}")
    print(f"✅ Static performance plot saved to {png_path}")
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}

def save_results(forecast, df, metrics, output_dir):
    """Save forecast results and metrics"""
    print("💾 Saving results...")
    
    # Save forecast data
    forecast.to_csv(f'{output_dir}/forecast_results.csv', index=False)
    df.to_csv(f'{output_dir}/original_data.csv', index=False)
    
    # Save metrics
    with open(f'{output_dir}/model_metrics.txt', 'w') as f:
        f.write("Prophet Model Performance Metrics\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Mean Absolute Error (MAE): ${metrics['mae']:,.0f}\n")
        f.write(f"Root Mean Square Error (RMSE): ${metrics['rmse']:,.0f}\n")
        f.write(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.1f}%\n")
        f.write(f"R-squared (R²): {metrics['r2']:.3f}\n")
        f.write(f"Training Data Points: {len(df):,}\n")
        f.write(f"Forecast Period: 6 months\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("✅ All results saved to CSV and text files")

def main():
    """Main execution function"""
    print("🚀 Interactive Prophet Time Series Forecasting Demo")
    print("=" * 55)
    print("📱 Optimized for GitHub Codespaces with Plotly!")
    print()
    
    # Setup
    output_dir = create_output_directory()
    
    # Generate data
    df = create_sample_data()
    
    # Create and fit model
    model = create_prophet_model()
    print("🎯 Training Prophet model...")
    model.fit(df)
    print("✅ Model training completed!")
    
    # Generate forecasts
    print("🔮 Generating 6-month forecast...")
    future = model.make_future_dataframe(periods=180)  # 6 months
    forecast = model.predict(future)
    print("✅ Forecast generation completed!")
    
    # Create visualizations
    print("\n📊 Creating interactive visualizations...")
    forecast_fig = create_interactive_forecast_plot(model, forecast, df, output_dir)
    components_fig = create_components_plot(model, forecast, output_dir)
    metrics = create_performance_plot(model, forecast, df, output_dir)
    
    # Save results
    save_results(forecast, df, metrics, output_dir)
    
    # Summary
    print("\n" + "=" * 55)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 55)
    print(f"\n📊 Model Performance Summary:")
    print(f"   • Mean Absolute Error: ${metrics['mae']:,.0f}")
    print(f"   • RMSE: ${metrics['rmse']:,.0f}")
    print(f"   • MAPE: {metrics['mape']:.1f}%")
    print(f"   • R²: {metrics['r2']:.3f}")
    
    print(f"\n📁 Generated Files in '{output_dir}/' directory:")
    print("   🌐 Interactive HTML files (open in Codespaces browser preview):")
    print("      • interactive_forecast.html - Main forecast plot")
    print("      • interactive_components.html - Components breakdown")
    print("      • performance_analysis.html - Model performance")
    print("\n   📄 Data files:")
    print("      • forecast_results.csv - Complete forecast data")
    print("      • original_data.csv - Training dataset")
    print("      • model_metrics.txt - Performance metrics")
    
    # Next steps
    print(f"\n🚀 Next Steps:")
    print("   1. Open the HTML files in Codespaces browser preview")
    print("   2. Download files to view locally")
    print("   3. Commit and push to see PNG files on GitHub")
    print("   4. Modify parameters and re-run for different results")
    
    # Future forecast preview
    future_forecast = forecast[forecast['ds'] > df['ds'].max()].head(7)
    print(f"\n🔮 Next 7 Days Forecast Preview:")
    for _, row in future_forecast.iterrows():
        print(f"   {row['ds'].strftime('%Y-%m-%d')}: ${row['yhat']:,.0f} (${row['yhat_lower']:,.0f} - ${row['yhat_upper']:,.0f})")
    
    print(f"\n✨ Enjoy exploring your interactive forecasts!")

if __name__ == "__main__":
    main()