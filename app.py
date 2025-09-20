# Simple Prophet Display App for Hugging Face Spaces
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import StringIO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Prophet Forecasting Results",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .tab-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .last-updated {
        background: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data_from_github():
    """Load forecast data directly from GitHub repository"""
    github_base = "https://raw.githubusercontent.com/ahtalebi/prophet-demo/main/outputs/"
    
    try:
        # Load forecast data
        forecast_url = github_base + "forecast_results.csv"
        response = requests.get(forecast_url, timeout=10)
        if response.status_code == 200:
            forecast_df = pd.read_csv(StringIO(response.text))
            forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
        else:
            return None, None, None, "Failed to load forecast data"
        
        # Load original data  
        original_url = github_base + "original_data.csv"
        response = requests.get(original_url, timeout=10)
        if response.status_code == 200:
            original_df = pd.read_csv(StringIO(response.text))
            original_df['ds'] = pd.to_datetime(original_df['ds'])
        else:
            return forecast_df, None, None, "Failed to load original data"
            
        # Load metrics
        metrics_url = github_base + "model_metrics.txt"
        metrics_text = "Metrics not available"
        try:
            response = requests.get(metrics_url, timeout=10)
            if response.status_code == 200:
                metrics_text = response.text
        except:
            pass
            
        return forecast_df, original_df, metrics_text, None
        
    except Exception as e:
        return None, None, None, f"Error loading data: {str(e)}"

def create_forecast_plot(forecast_df, original_df):
    """Create the main forecast visualization"""
    fig = go.Figure()
    
    # Historical data
    if original_df is not None:
        fig.add_trace(go.Scatter(
            x=original_df['ds'],
            y=original_df['y'],
            mode='markers',
            name='Historical Data',
            marker=dict(size=3, color='#2E86C1', opacity=0.7),
            hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> %{y:,.0f}<extra></extra>'
        ))
    
    # Future predictions
    if forecast_df is not None:
        future_data = forecast_df[forecast_df['ds'] > original_df['ds'].max()] if original_df is not None else forecast_df
        
        fig.add_trace(go.Scatter(
            x=future_data['ds'],
            y=future_data['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='#E74C3C', width=3),
            hovertemplate='<b>Date:</b> %{x}<br><b>Forecast:</b> %{y:,.0f}<extra></extra>'
        ))
        
        # Confidence intervals
        if 'yhat_upper' in future_data.columns and 'yhat_lower' in future_data.columns:
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
                hovertemplate='<b>Date:</b> %{x}<br><b>Range:</b> %{y:,.0f}<extra></extra>'
            ))
        
        # Model fit (historical)
        if original_df is not None:
            historical_forecast = forecast_df[forecast_df['ds'] <= original_df['ds'].max()]
            fig.add_trace(go.Scatter(
                x=historical_forecast['ds'],
                y=historical_forecast['yhat'],
                mode='lines',
                name='Model Fit',
                line=dict(color='#F39C12', width=2, dash='dot'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Fit:</b> %{y:,.0f}<extra></extra>'
            ))
    
    fig.update_layout(
        title='Prophet Time Series Forecast',
        xaxis_title='Date',
        yaxis_title='Value',
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

def create_components_plot(forecast_df):
    """Create components breakdown visualization"""
    if forecast_df is None:
        return go.Figure()
        
    # Check which components are available
    components = []
    component_names = []
    colors = ['#2E86C1', '#E74C3C', '#27AE60', '#8E44AD', '#F39C12']
    
    if 'trend' in forecast_df.columns:
        components.append('trend')
        component_names.append('Overall Trend')
    if 'yearly' in forecast_df.columns:
        components.append('yearly') 
        component_names.append('Yearly Seasonality')
    if 'weekly' in forecast_df.columns:
        components.append('weekly')
        component_names.append('Weekly Seasonality')
    if 'holidays' in forecast_df.columns:
        components.append('holidays')
        component_names.append('Holiday Effects')
    if 'monthly' in forecast_df.columns:
        components.append('monthly')
        component_names.append('Monthly Seasonality')
    
    if not components:
        st.warning("No component data available in the forecast.")
        return go.Figure()
    
    # Create subplots
    fig = make_subplots(
        rows=len(components), cols=1,
        subplot_titles=component_names,
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    for i, component in enumerate(components):
        fig.add_trace(
            go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df[component],
                mode='lines',
                name=component_names[i],
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'<b>Date:</b> %{{x}}<br><b>{component_names[i]}:</b> %{{y:,.2f}}<extra></extra>'
            ),
            row=i+1, col=1
        )
    
    fig.update_layout(
        title='Prophet Model Components Breakdown',
        height=150 * len(components) + 100,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Prophet Forecasting Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Automated forecasting results updated from GitHub repository**")
    
    # Load data
    with st.spinner("🔄 Loading latest forecast data from GitHub..."):
        forecast_df, original_df, metrics_text, error = load_data_from_github()
    
    if error:
        st.error(f"❌ {error}")
        st.info("💡 Make sure your GitHub repository has the required CSV files in the outputs folder.")
        st.stop()
    
    # Last updated info
    if forecast_df is not None:
        last_date = forecast_df['ds'].max().strftime('%Y-%m-%d')
        st.markdown(f"""
        <div class="last-updated">
            📅 <strong>Latest forecast data through:</strong> {last_date}
            <br>
            🔄 <strong>Auto-refreshes:</strong> Every 5 minutes
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📈 **Forecast**", "🔍 **Components**", "📊 **Metrics**"])
    
    with tab1:
        st.markdown('<h2 class="tab-header">Time Series Forecast</h2>', unsafe_allow_html=True)
        
        if forecast_df is not None:
            # Create and display forecast plot
            forecast_fig = create_forecast_plot(forecast_df, original_df)
            st.plotly_chart(forecast_fig, use_container_width=True)
            
            # Show some key stats
            if original_df is not None:
                future_data = forecast_df[forecast_df['ds'] > original_df['ds'].max()]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Historical Data Points", f"{len(original_df):,}")
                with col2:
                    st.metric("🔮 Forecast Points", f"{len(future_data):,}")
                with col3:
                    if len(future_data) > 0:
                        forecast_avg = future_data['yhat'].mean()
                        st.metric("📈 Avg Future Value", f"{forecast_avg:,.0f}")
        else:
            st.warning("⚠️ No forecast data available")
    
    with tab2:
        st.markdown('<h2 class="tab-header">Model Components</h2>', unsafe_allow_html=True)
        
        if forecast_df is not None:
            # Create and display components plot
            components_fig = create_components_plot(forecast_df)
            st.plotly_chart(components_fig, use_container_width=True)
            
            st.info("📝 **Components Explanation:**\n"
                   "- **Trend**: Overall direction of the data over time\n"
                   "- **Yearly Seasonality**: Annual patterns and cycles\n"
                   "- **Weekly Seasonality**: Weekly recurring patterns\n"
                   "- **Holiday Effects**: Impact of holidays on the forecast")
        else:
            st.warning("⚠️ No components data available")
    
    with tab3:
        st.markdown('<h2 class="tab-header">Model Performance</h2>', unsafe_allow_html=True)
        
        if metrics_text and metrics_text != "Metrics not available":
            # Display metrics in a nice format
            st.text_area("📊 Model Performance Metrics", metrics_text, height=300)
        else:
            st.warning("⚠️ No metrics data available")
        
        # Additional info
        st.markdown("---")
        st.markdown("### 🔄 How This Works")
        st.markdown("""
        1. **Local Development**: You update `prophet_demo.py` locally
        2. **GitHub Push**: Changes are pushed to your GitHub repository
        3. **GitHub Actions**: Automatically runs Prophet forecasting
        4. **Auto-Update**: This dashboard loads the latest results
        5. **Live Display**: Updated plots appear here automatically!
        """)
        
        # Refresh button
        if st.button("🔄 Force Refresh Data", type="secondary"):
            st.cache_data.clear()
            st.rerun()

if __name__ == "__main__":
    main()
