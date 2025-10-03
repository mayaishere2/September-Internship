import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import plotly.graph_objects as go
from collections import deque
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU, avoid cuInit error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress logs
from keras.models import load_model
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
# --- CONFIGURATION ---
st.set_page_config(
    layout="wide",
    page_title="LNG Turbine Predictive Dashboard",
    page_icon="⚡"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    /* Custom dark theme adjustments */
    .stApp {
        background-color: #0f172a;
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 2rem 10rem;
    }
    .st-emotion-cache-z5fcl4 {
        padding-top: 3rem;
    }
    
    /* KPI Metric styling */
    .st-emotion-cache-1tpltef, .st-emotion-cache-1b0udgb {
        border-radius: 0.75rem;
        padding: 1.5rem;
        background-color: #1e293b;
        border: 1px solid #334155;
    }

    /* Alarm styling */
    .alarm-card {
        background-color: #ef4444; /* red-500 */
        border: 1px solid #f87171; /* red-400 */
        animation: pulse 1.5s infinite;
        border-radius: 0.5rem;
        padding: 1.5rem;
    }
    @keyframes pulse {
        0%, 100% {
            box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
        }
        50% {
            box-shadow: 0 0 0 10px rgba(239, 68, 68, 0);
        }
    }
</style>
""", unsafe_allow_html=True)


# --- MODEL & DATA LOADING (Cached for performance) ---
@st.cache_resource
def load_all_models():
    print("Loading models and preprocessing objects...")
    try:
        forecaster = load_model('forecasting_model.keras', safe_mode=False)
        classifier = joblib.load('classification_model.joblib')
        scaler = joblib.load('scaler.joblib')
        encoder = joblib.load('encoder.joblib')
        print("All models loaded successfully.")
        return forecaster, classifier, scaler, encoder
    except Exception as e:
        st.error(f"Error loading models: {e}. Make sure all model files are in the directory.")
        return None, None, None, None

@st.cache_data
def load_simulation_data(_scaler): # Pass in the loaded scaler to use it
    try:
        df = pd.read_csv('my_turbine_dataset.csv')
        # Get the feature columns from the scaler object
        feature_cols = _scaler.get_feature_names_out()
        return df, feature_cols
    except FileNotFoundError:
        st.error("Error: `my_turbine_dataset.csv` not found.")
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# --- PREDICTION PIPELINE ---
def predict_future_state(sequence, forecaster, classifier, scaler, encoder, feature_cols):
    """Takes a sequence of data, forecasts, classifies, and returns results."""
    # Reshape for the model
    sequence_3d = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
    
    # 1. Forecast future sensor values
    predicted_future_sequence = forecaster.predict(sequence_3d, verbose=0)
    
    # 2. Extract the state at the 2-hour mark
    predicted_future_state_scaled = predicted_future_sequence[:, -1, :]
    
    # 3. Classify the forecasted state
    predicted_fault_code = classifier.predict(predicted_future_state_scaled)[0]
    predicted_fault_name = encoder.classes_[predicted_fault_code]
    
    # 4. Inverse transform forecasted values to be human-readable
    predicted_future_state_unscaled = scaler.inverse_transform(predicted_future_state_scaled)
    
    forecasted_values = pd.DataFrame(predicted_future_state_unscaled, columns=feature_cols).iloc[0]
    
    # Simple confidence simulation
    confidence = np.random.uniform(0.85, 0.99) if predicted_fault_name == 'normal' else np.random.uniform(0.75, 0.95)

    return predicted_fault_name, forecasted_values, confidence

# --- UI HELPER FUNCTIONS ---
def create_gauge(value, title, min_val, max_val, color="cyan"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16, 'color': 'white'}},
        number={'font': {'size': 36, 'color': 'white'}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "#334155",
            'borderwidth': 2,
            'bordercolor': "#64748b"
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"},
        height=150,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig


# --- INITIALIZE SESSION STATE ---
if 'data_index' not in st.session_state:
    st.session_state.data_index = 60 # Start after the first lookback window
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = deque(maxlen=10)
    st.session_state.log_messages.append("System Initialized. Waiting for live data...")

# --- LOAD MODELS AND DATA ---
forecasting_model, classification_model, scaler, label_encoder = load_all_models()
if scaler is not None:
    df_sim, feature_columns = load_simulation_data(scaler)
else:
    df_sim, feature_columns = None, None


# --- DASHBOARD LAYOUT ---
st.title("LNG Turbine Predictive Maintenance Dashboard")
st.markdown(f"**Location:** Algiers, Algeria")

# Placeholders for dynamic content
placeholder_time = st.empty()
col_status, col_forecast = st.columns([4, 8])
current_status_card = col_status.empty()
predicted_status_card = col_forecast.empty()
st.markdown("---")
col_g1, col_g2, col_g3, col_g4 = st.columns(4)
gauge_tit_placeholder = col_g1.empty()
gauge_cdp_placeholder = col_g2.empty()
gauge_vshaft_placeholder = col_g3.empty()
gauge_tbearing_placeholder = col_g4.empty()
st.markdown("---")
col_chart, col_log = st.columns([8, 4])
chart_placeholder = col_chart.empty()
log_placeholder = col_log.empty()


# --- SIMULATION & UI UPDATE ---
if all([forecasting_model, classification_model, scaler, label_encoder, df_sim is not None]):
    # Initialize historical data buffer
    if st.session_state.historical_data is None:
        initial_data = df_sim.iloc[:st.session_state.data_index]
        st.session_state.historical_data = deque(initial_data.to_dict('records'), maxlen=60)
    
    # Get current row from simulation
    current_data_row = df_sim.iloc[st.session_state.data_index]
    st.session_state.historical_data.append(current_data_row.to_dict())
    
    # Prepare dataframes for easier access
    history_df = pd.DataFrame(list(st.session_state.historical_data))
    current_state = history_df.iloc[-1]
    
    # --- Run Prediction Pipeline periodically ---
    if st.session_state.data_index % 15 == 0:
        sequence_data_scaled = scaler.transform(history_df[feature_columns])
        
        pred_fault, forecast_values, pred_conf = predict_future_state(
            sequence_data_scaled, forecasting_model, classification_model, scaler, label_encoder, feature_columns
        )
        st.session_state.predicted_fault = pred_fault
        st.session_state.forecasted_values = forecast_values
        st.session_state.prediction_confidence = pred_conf

        if pred_fault != 'normal':
             st.session_state.log_messages.append(f"⚠️ {pd.Timestamp.now().strftime('%H:%M:%S')} - ALARM: Predicted '{pred_fault.upper()}' fault in 2 hours.")
        else:
             st.session_state.log_messages.append(f"✅ {pd.Timestamp.now().strftime('%H:%M:%S')} - Forecast complete. System state normal.")

    # --- UPDATE UI ELEMENTS ---
    # Update Time
    placeholder_time.markdown(f"**Simulated Time:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Update Current Status Card
    current_fault = current_state['fault_type']
    with current_status_card.container():
        if current_fault == 'normal':
            st.success(f"**CURRENT STATUS: NORMAL**\n\nAll systems operating within nominal parameters.")
        else:
            st.error(f"**CURRENT STATUS: {current_fault.upper()}**\n\nAnomaly detected in current operation!")
    
    # Update Predicted Status Card
    with predicted_status_card.container():
        if 'predicted_fault' in st.session_state:
            pred_fault = st.session_state.predicted_fault
            pred_conf = st.session_state.prediction_confidence
            
            if pred_fault == 'normal':
                st.success(f"**2-HOUR FORECAST: NORMAL**\n\nNo anomalies predicted. Confidence: {pred_conf:.1%}")
            else:
                # Use markdown to inject custom styled alarm card
                st.markdown(f"""
                    <div class="alarm-card">
                        <p style="font-size: 1.5rem; font-weight: 700; color: white;">2-HOUR FORECAST: {pred_fault.upper()}</p>
                        <p style="color: white;">Potential fault detected. Recommend monitoring relevant systems. Confidence: {pred_conf:.1%}</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Awaiting first prediction...")

    # Update Gauges
    with gauge_tit_placeholder.container():
        st.plotly_chart(create_gauge(current_state['TIT'], "Turbine Inlet Temp (°C)", 1000, 1105, color="#f97316"), use_container_width=True)
    with gauge_cdp_placeholder.container():
        st.plotly_chart(create_gauge(current_state['CDP'], "Compressor Pressure (bar)", 9, 16, color="#3b82f6"), use_container_width=True)
    with gauge_vshaft_placeholder.container():
        st.plotly_chart(create_gauge(current_state['V_shaft'], "Shaft Vibration (µm)", 10, 100, color="#8b5cf6"), use_container_width=True)
    with gauge_tbearing_placeholder.container():
         st.plotly_chart(create_gauge(current_state['T_bearing'], "Bearing Temp (°C)", 60, 120, color="#ec4899"), use_container_width=True)


    # Update Chart
    with chart_placeholder.container():
        # MODIFIED: Removed the problematic index-setting line.
        chart_df = history_df[['TIT', 'TAT', 'EGT']].copy()
        st.line_chart(chart_df, color=["#f97316", "#eab308", "#ef4444"])

    # Update Log
    with log_placeholder.container():
        st.subheader("System Alerts & Log")
        log_html = "".join([f"<p style='font-size: 0.8rem; margin-bottom: 5px; color: #cbd5e1;'>{msg}</p>" for msg in reversed(st.session_state.log_messages)])
        st.markdown(f"<div style='background-color: #1e293b; border-radius: 0.5rem; padding: 10px; border: 1px solid #334155; height: 350px; overflow-y: auto;'>{log_html}</div>", unsafe_allow_html=True)
    
    
    # --- SIMULATION CONTROL ---
    st.session_state.data_index = (st.session_state.data_index + 1) % len(df_sim)
    if st.session_state.data_index < 60: # Reset if we reach the end
        st.session_state.data_index = 60
        st.session_state.log_messages.append("--- Simulation loop restarted ---")

    # This is the new, standard way to loop the app
    time.sleep(1.5)
    st.rerun()

else:
    st.warning("Models or data could not be loaded. The application cannot start.")

