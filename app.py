# Streamlit Prophet Demo for Hugging Face Spaces
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Prophet Forecasting Demo",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def create_sample_data(data_points=1000, noise_level=100):
    """Create realistic e-commerce sales data"""
    np.random.seed(42)
    
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2024, 8, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Take only the requested number of points
    if len(dates) > data_points:
        dates = dates[-data_points:]
    
    n_days = len(dates)
    
    # Create realistic sales components
    trend = np.linspace(1000, 3500, n_days)
    yearly_seasonal = 400 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    weekly_seasonal = 200 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    monthly_effect = 150 * np.sin(2 * np.pi * np.arange(n_days) / 30.44)
    noise = np.random.normal(0, noise_level, n_days)
    
    # Special events (Black Friday, Christmas)
    special_events = np.zeros(n_days)
    for i, date in enumerate(dates):
        if date.month == 11 and date.day >= 24 and date.day <= 28:  # Black Friday week
            special_events[i] = 500
        elif date.month == 12 and date.day >= 15:  # Christmas season
            special_events[i] = 300
    
    # Combine all components
    sales = trend + yearly_seasonal + weekly_seasonal + monthly_effect + special_events + noise
    sales = np.maximum(sales, 100)  # No negative sales
    
    df = pd.DataFrame({
        'ds': dates,
        'y': sales
    })
    
    return df

@st.cache_resource
def create_prophet_model(growth_type, seasonality_mode, changepoint_scale, seasonality_scale):
    """Create and configure Prophet model"""
    model = Prophet(
        growth=growth_type,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_scale,
        seasonality_prior_scale=seasonality_scale,
        holidays_prior_scale=10.0,
        interval_width=0.8
    )
    
    # Add US holidays
    model.add_country_holidays(country_name='US')
    
    # Add custom seasonality
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    return model

def create_forecast_plot(model, forecast, df, show_components=True):
    """Create interactive forecast plot"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='markers',
        name='Historical Data',
        marker=dict(size=4, color='#2E86C1', opacity=0.7),
        hovertemplate='<b>Date:</b> %{x}<br><b>Sales:</b> $%{y:,.0f}<extra></extra>'
    ))
    
    # Forecast line
    future_data = forecast[forecast['ds'] > df['ds'].max()]
    fig.add_trace(go.Scatter(
        x=future_data['ds'],
        y=future_data['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#E74C3C', width=3),
        hovertemplate='<b>Date:</b> %{x}<br><b>Forecast:</b> $%{y:,.0f}<extra></extra>'
    ))
    
    # Confidence interval
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
        hovertemplate='<b>Date:</b> %{x}<br><b>Lower Bound:</b> $%{y:,.0f}<extra></extra>'
    ))
    
    # Model fit for historical data
    historical_forecast = forecast[forecast['ds'] <= df['ds'].max()]
    fig.add_trace(go.Scatter(
        x=historical_forecast['ds'],
        y=historical_forecast['yhat'],
        mode='lines',
        name='Model Fit',
        line=dict(color='#F39C12', width=2, dash='dot'),
        hovertemplate='<b>Date:</b> %{x}<br><b>Model Fit:</b> $%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Prophet Time Series Forecast',
        xaxis_title='Date',
        yaxis_title='Sales (USD)',
        hovermode='x unified',
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        )
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def create_components_plot(forecast):
    """Create components breakdown plot"""
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=['Overall Trend', 'Yearly Seasonality', 'Weekly Seasonality', 'Holiday Effects'],
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # Trend
    fig.add_trace(
        go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', 
                  name='Trend', line=dict(color='#2E86C1', width=2)),
        row=1, col=1
    )
    
    # Yearly seasonality
    if 'yearly' in forecast.columns:
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['yearly'], mode='lines',
                      name='Yearly', line=dict(color='#E74C3C', width=2)),
            row=2, col=1
        )
    
    # Weekly seasonality
    if 'weekly' in forecast.columns:
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['weekly'], mode='lines',
                      name='Weekly', line=dict(color='#27AE60', width=2)),
            row=3, col=1
        )
    
    # Holiday effects
    if 'holidays' in forecast.columns:
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['holidays'], mode='lines',
                      name='Holidays', line=dict(color='#8E44AD', width=2)),
            row=4, col=1
        )
    
    fig.update_layout(
        title='Prophet Model Components Breakdown',
        height=800,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def calculate_metrics(model, forecast, df):
    """Calculate model performance metrics"""
    # Get training predictions
    train_forecast = forecast[forecast['ds'] <= df['ds'].max()]
    merged = df.merge(train_forecast[['ds', 'yhat']], on='ds')
    
    # Calculate metrics
    mae = np.mean(np.abs(merged['y'] - merged['yhat']))
    rmse = np.sqrt(np.mean((merged['y'] - merged['yhat']) ** 2))
    mape = np.mean(np.abs((merged['y'] - merged['yhat']) / merged['y'])) * 100
    r2 = np.corrcoef(merged['y'], merged['yhat'])[0, 1] ** 2
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2
    }

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🔮 Prophet Time Series Forecasting Demo</h1>', unsafe_allow_html=True)
    st.markdown("**Interactive demonstration of Facebook's Prophet library with realistic e-commerce data**")
    
    # Sidebar controls
    st.sidebar.header("🛠️ Model Configuration")
    
    # Data parameters
    st.sidebar.subheader("📊 Data Parameters")
    data_points = st.sidebar.slider("Training Data Points", 500, 1500, 1000, 50)
    noise_level = st.sidebar.slider("Noise Level", 50, 200, 100, 10)
    
    # Model parameters
    st.sidebar.subheader("🤖 Model Parameters")
    growth_type = st.sidebar.selectbox("Growth Type", ["linear", "logistic"], index=0)
    seasonality_mode = st.sidebar.selectbox("Seasonality Mode", ["additive", "multiplicative"], index=0)
    changepoint_scale = st.sidebar.slider("Trend Flexibility", 0.001, 0.5, 0.05, 0.001, format="%.3f")
    seasonality_scale = st.sidebar.slider("Seasonality Strength", 0.01, 50.0, 10.0, 0.1)
    
    # Forecast parameters
    st.sidebar.subheader("🔮 Forecast Parameters")
    forecast_days = st.sidebar.slider("Days to Forecast", 30, 365, 180, 30)
    
    # Generate button
    if st.sidebar.button("🚀 Generate Forecast", type="primary"):
        with st.spinner("🔄 Generating forecast... This may take a moment."):
            
            # Create data
            df = create_sample_data(data_points, noise_level)
            
            # Create and fit model
            model = create_prophet_model(growth_type, seasonality_mode, changepoint_scale, seasonality_scale)
            model.fit(df)
            
            # Generate forecast
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            
            # Calculate metrics
            metrics = calculate_metrics(model, forecast, df)
            
            # Store in session state
            st.session_state['df'] = df
            st.session_state['forecast'] = forecast
            st.session_state['model'] = model
            st.session_state['metrics'] = metrics
            st.session_state['forecast_generated'] = True
        
        st.success("✅ Forecast generated successfully!")
    
    # Display results if forecast has been generated
    if st.session_state.get('forecast_generated', False):
        df = st.session_state['df']
        forecast = st.session_state['forecast']
        model = st.session_state['model']
        metrics = st.session_state['metrics']
        
        # Performance metrics
        st.header("📊 Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>MAE</h3>
                <h2>${metrics['MAE']:,.0f}</h2>
                <p>Mean Absolute Error</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>RMSE</h3>
                <h2>${metrics['RMSE']:,.0f}</h2>
                <p>Root Mean Square Error</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>MAPE</h3>
                <h2>{metrics['MAPE']:.1f}%</h2>
                <p>Mean Absolute % Error</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>R²</h3>
                <h2>{metrics['R²']:.3f}</h2>
                <p>Coefficient of Determination</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Main forecast plot
        st.header("📈 Forecast Visualization")
        forecast_fig = create_forecast_plot(model, forecast, df)
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Components plot
        st.header("🔍 Components Analysis")
        components_fig = create_components_plot(forecast)
        st.plotly_chart(components_fig, use_container_width=True)
        
        # Future predictions table
        st.header("🔮 Future Predictions Preview")
        future_data = forecast[forecast['ds'] > df['ds'].max()].head(14)
        
        # Format the dataframe for display
        future_display = future_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        future_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
        future_display['Date'] = future_display['Date'].dt.strftime('%Y-%m-%d')
        future_display['Forecast'] = future_display['Forecast'].apply(lambda x: f'${x:,.0f}')
        future_display['Lower Bound'] = future_display['Lower Bound'].apply(lambda x: f'${x:,.0f}')
        future_display['Upper Bound'] = future_display['Upper Bound'].apply(lambda x: f'${x:,.0f}')
        
        st.dataframe(future_display, use_container_width=True)
        
        # Download data
        st.header("📁 Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # Download forecast data
            forecast_csv = forecast.to_csv(index=False)
            st.download_button(
                label="📊 Download Forecast Data (CSV)",
                data=forecast_csv,
                file_name="prophet_forecast.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download original data
            original_csv = df.to_csv(index=False)
            st.download_button(
                label="📋 Download Training Data (CSV)",
                data=original_csv,
                file_name="training_data.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👈 Configure your parameters in the sidebar and click 'Generate Forecast' to begin!")
        
        # Show sample info
        st.header("ℹ️ About This Demo")
        
        st.markdown("""
        This interactive demo showcases **Facebook's Prophet** library for time series forecasting:
        
        ### 🔧 Features
        - **Interactive Parameter Tuning**: Adjust model parameters in real-time
        - **Realistic Data**: Simulated e-commerce sales with seasonal patterns
        - **Professional Visualizations**: Interactive Plotly charts
        - **Performance Metrics**: MAE, RMSE, MAPE, and R² evaluation
        - **Components Analysis**: Breakdown of trend, seasonality, and holidays
        - **Data Export**: Download results for further analysis
        
        ### 📊 Model Capabilities
        - Automatic seasonality detection (yearly, weekly, monthly)
        - US holiday integration
        - Confidence interval estimation
        - Trend changepoint detection
        - Robust outlier handling
        
        ### 🚀 Getting Started
        1. Adjust parameters in the sidebar
        2. Click "Generate Forecast"  
        3. Explore the interactive visualizations
        4. Download your results
        
        **Perfect for**: Business forecasting, educational purposes, and portfolio demonstrations.
        """)

if __name__ == "__main__":
    main()
