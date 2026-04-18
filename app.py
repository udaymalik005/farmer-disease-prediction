"""
🌾 FARMER'S CROP DISEASE PREDICTION SYSTEM
Modern Streamlit Dashboard | AI/ML Powered
Research-grade application for agricultural disease detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="AgroSense AI | Crop Disease Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    /* Root variables */
    :root {
        --green-dark: #1a3a2a;
        --green-mid: #2d6a4f;
        --green-bright: #52b788;
        --green-light: #95d5b2;
        --amber: #f4a261;
        --red-alert: #e63946;
        --cream: #f8f5f0;
        --text-primary: #1a1a2e;
        --card-bg: rgba(255,255,255,0.85);
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0d2818 0%, #1a3a2a 30%, #0f3d20 60%, #1e4d35 100%);
        font-family: 'DM Sans', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d2818 0%, #1a3a2a 100%) !important;
        border-right: 1px solid rgba(82, 183, 136, 0.3);
    }
    section[data-testid="stSidebar"] * {
        color: #e8f5e9 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label {
        color: #95d5b2 !important;
        font-size: 0.82rem !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Headings */
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(82,183,136,0.25);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        backdrop-filter: blur(12px);
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(82,183,136,0.6);
    }
    .metric-card .label {
        font-size: 0.78rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #95d5b2;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'Syne', sans-serif;
        color: #ffffff;
    }
    .metric-card .sub {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.5);
        margin-top: 0.2rem;
    }

    /* Prediction result */
    .result-healthy {
        background: linear-gradient(135deg, rgba(82,183,136,0.2), rgba(82,183,136,0.05));
        border: 2px solid #52b788;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }
    .result-disease {
        background: linear-gradient(135deg, rgba(230,57,70,0.2), rgba(230,57,70,0.05));
        border: 2px solid #e63946;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }
    .result-moderate {
        background: linear-gradient(135deg, rgba(244,162,97,0.2), rgba(244,162,97,0.05));
        border: 2px solid #f4a261;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }

    /* Header hero */
    .hero-banner {
        background: linear-gradient(135deg, rgba(82,183,136,0.15) 0%, rgba(45,106,79,0.2) 100%);
        border: 1px solid rgba(82,183,136,0.3);
        border-radius: 24px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
    }
    .hero-sub {
        color: #95d5b2;
        font-size: 1.05rem;
        margin-top: 0.5rem;
    }

    /* Section headers */
    .section-header {
        font-family: 'Syne', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #52b788;
        letter-spacing: -0.01em;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(82,183,136,0.2);
        margin-bottom: 1.2rem;
    }

    /* Disease badge */
    .disease-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Treatment box */
    .treatment-box {
        background: rgba(82,183,136,0.08);
        border-left: 3px solid #52b788;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
        color: #e8f5e9;
        font-size: 0.88rem;
    }

    /* Override Streamlit defaults */
    .stMetric {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 0.8rem;
        border: 1px solid rgba(82,183,136,0.2);
    }
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(82,183,136,0.2);
        border-radius: 12px;
        padding: 0.8rem;
    }
    [data-testid="metric-container"] label {
        color: #95d5b2 !important;
        font-size: 0.8rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: white !important;
        font-family: 'Syne', sans-serif !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(82,183,136,0.2);
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #95d5b2 !important;
        font-family: 'DM Sans', sans-serif;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(82,183,136,0.2) !important;
        color: #ffffff !important;
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] div {
        background: #52b788 !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #2d6a4f, #52b788) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.7rem 2rem !important;
        width: 100%;
        transition: all 0.2s;
        letter-spacing: 0.02em;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(82,183,136,0.4) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d2818; }
    ::-webkit-scrollbar-thumb { background: #2d6a4f; border-radius: 3px; }

    /* Divider */
    hr { border-color: rgba(82,183,136,0.2) !important; }

    /* DataFrame */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* Alert/info boxes */
    .stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ─── CONSTANTS ─────────────────────────────────────────────────
DISEASE_INFO = {
    "Healthy": {
        "icon": "✅", "color": "#52b788", "severity": "None",
        "description": "Crop is in excellent health with no signs of disease.",
        "treatment": ["Continue regular crop monitoring", "Maintain balanced fertilization", "Practice crop rotation", "Regular field scouting"],
        "prevention": "Maintain good agronomic practices and regular monitoring."
    },
    "Leaf Blight": {
        "icon": "🍂", "color": "#e63946", "severity": "High",
        "description": "Fungal disease causing brown lesions on leaves, leading to defoliation.",
        "treatment": ["Apply Mancozeb 75% WP @ 2.5 g/L water", "Remove and destroy infected leaves", "Improve field drainage", "Apply Copper Oxychloride spray"],
        "prevention": "Use resistant varieties, avoid overhead irrigation, ensure proper spacing."
    },
    "Powdery Mildew": {
        "icon": "🌫️", "color": "#f4a261", "severity": "Moderate",
        "description": "White powdery fungal growth on leaf surfaces reducing photosynthesis.",
        "treatment": ["Spray Sulfur 80% WP @ 2 g/L", "Apply Triadimefon 25% WP", "Remove infected plant parts", "Neem oil spray @ 5 ml/L"],
        "prevention": "Maintain proper plant spacing, avoid excess nitrogen, use disease-free seeds."
    },
    "Root Rot": {
        "icon": "🫚", "color": "#e63946", "severity": "Severe",
        "description": "Soil-borne pathogen causing root decay, wilting, and plant death.",
        "treatment": ["Drench soil with Carbendazim @ 1 g/L", "Improve soil drainage immediately", "Apply Trichoderma viride biocontrol", "Reduce irrigation frequency"],
        "prevention": "Ensure good drainage, avoid waterlogging, use treated seeds."
    },
    "Rust Disease": {
        "icon": "🟤", "color": "#f4a261", "severity": "Moderate",
        "description": "Fungal pustules on leaves reducing yield by 20-40%.",
        "treatment": ["Apply Propiconazole 25% EC @ 1 ml/L", "Spray Hexaconazole @ 1 ml/L", "Remove infected stubble", "Apply Mancozeb preventively"],
        "prevention": "Plant resistant varieties, timely sowing, avoid dense planting."
    },
    "Bacterial Wilt": {
        "icon": "💧", "color": "#e63946", "severity": "Severe",
        "description": "Bacterial infection causing sudden wilting and plant collapse.",
        "treatment": ["No chemical cure available", "Remove and destroy infected plants", "Copper-based bactericides preventively", "Soil fumigation if severe"],
        "prevention": "Use certified disease-free seeds, rotate crops, control insect vectors."
    },
    "Downy Mildew": {
        "icon": "🌧️", "color": "#f4a261", "severity": "Moderate",
        "description": "Water mold causing yellow lesions with downy growth on leaf undersides.",
        "treatment": ["Apply Metalaxyl-M + Mancozeb", "Spray Cymoxanil @ 0.6 g/L", "Improve air circulation", "Reduce humidity around plants"],
        "prevention": "Avoid overcrowding, use resistant cultivars, reduce leaf wetness."
    },
    "Mosaic Virus": {
        "icon": "🦠", "color": "#e63946", "severity": "High",
        "description": "Viral infection causing mosaic patterns and stunted growth.",
        "treatment": ["No chemical cure for viruses", "Remove infected plants immediately", "Control aphid/whitefly vectors", "Apply mineral oil to deter vectors"],
        "prevention": "Use virus-indexed seeds, control insect vectors, practice field sanitation."
    },
    "Anthracnose": {
        "icon": "⚫", "color": "#f4a261", "severity": "Moderate",
        "description": "Fungal disease causing dark, sunken lesions on fruits and leaves.",
        "treatment": ["Apply Carbendazim 50% WP @ 1 g/L", "Spray Chlorothalonil @ 2 g/L", "Remove mummified fruits", "Improve orchard sanitation"],
        "prevention": "Prune infected branches, avoid overhead irrigation, clean tools."
    },
    "Fusarium Wilt": {
        "icon": "🥀", "color": "#e63946", "severity": "Severe",
        "description": "Soil-borne fungal disease blocking water transport, causing yellowing and death.",
        "treatment": ["Soil treatment with Carbendazim", "Apply Trichoderma harzianum biocontrol", "Use grafted resistant rootstocks", "Soil solarization"],
        "prevention": "Use resistant varieties, soil solarization, balanced fertilization."
    }
}

FEATURE_COLS = [
    'temperature_celsius', 'humidity_percent', 'rainfall_mm', 'soil_ph',
    'nitrogen_kg_ha', 'phosphorus_kg_ha', 'potassium_kg_ha',
    'wind_speed_kmh', 'sunlight_hours', 'leaf_wetness_hours',
    'field_area_hectares', 'days_after_sowing', 'prev_disease_history'
]


# ─── LOAD MODEL ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("models/disease_model.pkl")
    le = joblib.load("models/label_encoder.pkl")
    with open("models/model_metadata.json") as f:
        meta = json.load(f)
    return model, le, meta

@st.cache_data
def load_dataset():
    return pd.read_csv("data/crop_disease_dataset.csv")


# ─── PREDICTION ────────────────────────────────────────────────
def predict_disease(model, le, features_dict):
    df = pd.DataFrame([features_dict])[FEATURE_COLS]
    pred_enc = model.predict(df)[0]
    pred_proba = model.predict_proba(df)[0]
    pred_label = le.inverse_transform([pred_enc])[0]
    confidence = pred_proba[pred_enc]
    
    # Top 3 predictions
    top3_idx = np.argsort(pred_proba)[::-1][:3]
    top3 = [(le.inverse_transform([i])[0], pred_proba[i]) for i in top3_idx]
    
    return pred_label, confidence, top3


# ─── SIDEBAR ───────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1.5rem 0 1rem 0;'>
            <div style='font-size:3rem;'>🌾</div>
            <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:800; color:#52b788;'>AgroSense AI</div>
            <div style='font-size:0.75rem; color:#95d5b2; margin-top:4px;'>Crop Disease Intelligence</div>
        </div>
        <hr style='border-color:rgba(82,183,136,0.2);margin:0.5rem 0 1.2rem 0;'/>
        """, unsafe_allow_html=True)

        st.markdown("### 🔢 Field Parameters")
        st.caption("Enter your crop/field conditions below")

        crops = ["Wheat", "Rice", "Maize", "Tomato", "Potato", "Cotton", "Soybean", "Sugarcane", "Barley", "Sorghum"]
        crop = st.selectbox("Crop Type", crops)
        season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])

        st.markdown("**🌡️ Climate**")
        temp = st.slider("Temperature (°C)", 5.0, 45.0, 26.0, 0.5)
        humidity = st.slider("Humidity (%)", 10.0, 100.0, 65.0, 1.0)
        rainfall = st.slider("Rainfall (mm)", 0.0, 400.0, 100.0, 5.0)
        wind_speed = st.slider("Wind Speed (km/h)", 0.0, 40.0, 10.0, 0.5)
        sunlight = st.slider("Sunlight Hours", 2.0, 14.0, 7.0, 0.5)

        st.markdown("**🌱 Soil & Nutrients**")
        soil_ph = st.slider("Soil pH", 3.5, 9.0, 6.5, 0.1)
        nitrogen = st.slider("Nitrogen (kg/ha)", 0.0, 150.0, 60.0, 1.0)
        phosphorus = st.slider("Phosphorus (kg/ha)", 0.0, 120.0, 40.0, 1.0)
        potassium = st.slider("Potassium (kg/ha)", 0.0, 120.0, 50.0, 1.0)

        st.markdown("**🏡 Field Info**")
        leaf_wetness = st.slider("Leaf Wetness (hrs)", 0.0, 20.0, 5.0, 0.5)
        field_area = st.number_input("Field Area (hectares)", 0.1, 50.0, 2.0, 0.1)
        days_sowing = st.number_input("Days After Sowing", 1, 180, 45)
        prev_history = st.radio("Previous Disease History?", ["No", "Yes"])

        predict_btn = st.button("🔍 Predict Disease", use_container_width=True)

        st.markdown("""
        <hr style='margin:1.5rem 0 1rem 0;border-color:rgba(82,183,136,0.15);'/>
        <div style='font-size:0.7rem; color:rgba(149,213,178,0.6); text-align:center;'>
        Model Accuracy: 94.70% | RF Classifier<br/>
        Dataset: 5,000 samples | 10 diseases
        </div>
        """, unsafe_allow_html=True)

    features = {
        'temperature_celsius': temp,
        'humidity_percent': humidity,
        'rainfall_mm': rainfall,
        'soil_ph': soil_ph,
        'nitrogen_kg_ha': nitrogen,
        'phosphorus_kg_ha': phosphorus,
        'potassium_kg_ha': potassium,
        'wind_speed_kmh': wind_speed,
        'sunlight_hours': sunlight,
        'leaf_wetness_hours': leaf_wetness,
        'field_area_hectares': field_area,
        'days_after_sowing': int(days_sowing),
        'prev_disease_history': 1 if prev_history == "Yes" else 0
    }
    return features, predict_btn, crop, season


# ─── CHARTS ────────────────────────────────────────────────────
def plot_disease_distribution(df):
    counts = df['disease_label'].value_counts().reset_index()
    counts.columns = ['Disease', 'Count']
    fig = px.bar(
        counts, x='Count', y='Disease', orientation='h',
        color='Count', color_continuous_scale=['#2d6a4f', '#52b788', '#95d5b2'],
        title=''
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8f5e9', family='DM Sans'),
        coloraxis_showscale=False,
        margin=dict(l=0, r=10, t=10, b=0),
        xaxis=dict(gridcolor='rgba(82,183,136,0.1)', title=''),
        yaxis=dict(gridcolor='rgba(82,183,136,0.1)', title='')
    )
    fig.update_traces(marker_line_width=0)
    return fig

def plot_feature_importance(meta):
    fi = meta['feature_importance']
    labels = [k.replace('_', ' ').title() for k in fi.keys()]
    vals = list(fi.values())
    fig = go.Figure(go.Bar(
        x=vals[:8], y=labels[:8], orientation='h',
        marker=dict(
            color=vals[:8],
            colorscale=[[0, '#2d6a4f'], [0.5, '#52b788'], [1, '#95d5b2']],
            line=dict(width=0)
        )
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8f5e9', family='DM Sans'),
        margin=dict(l=0, r=10, t=10, b=0),
        xaxis=dict(gridcolor='rgba(82,183,136,0.1)', title='Importance Score'),
        yaxis=dict(gridcolor='rgba(82,183,136,0.1)', title='')
    )
    return fig

def plot_scatter(df, x, y, color='disease_label'):
    fig = px.scatter(
        df.sample(min(800, len(df))), x=x, y=y, color=color,
        color_discrete_sequence=px.colors.qualitative.Set3,
        opacity=0.7
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8f5e9', family='DM Sans'),
        margin=dict(l=0, r=10, t=10, b=0),
        xaxis=dict(gridcolor='rgba(82,183,136,0.08)'),
        yaxis=dict(gridcolor='rgba(82,183,136,0.08)'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10))
    )
    return fig

def plot_correlation(df):
    num_cols = ['temperature_celsius', 'humidity_percent', 'rainfall_mm', 'soil_ph',
                'nitrogen_kg_ha', 'phosphorus_kg_ha', 'potassium_kg_ha', 'yield_loss_percent']
    corr = df[num_cols].corr()
    labels = [c.replace('_celsius', '').replace('_percent', '%').replace('_mm', ' mm')
              .replace('_kg_ha', ' kg/ha').replace('_', ' ').title() for c in num_cols]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale=[[0, '#0d2818'], [0.5, '#2d6a4f'], [1, '#52b788']],
        text=np.round(corr.values, 2), texttemplate="%{text}",
        showscale=True
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8f5e9', family='DM Sans', size=10),
        margin=dict(l=0, r=0, t=10, b=0), height=380
    )
    return fig

def plot_yield_loss(df):
    avg = df[df['disease_label'] != 'Healthy'].groupby('disease_label')['yield_loss_percent'].mean().sort_values(ascending=False)
    fig = px.bar(
        x=avg.index, y=avg.values,
        color=avg.values, color_continuous_scale=['#2d6a4f', '#f4a261', '#e63946'],
        labels={'x': 'Disease', 'y': 'Avg Yield Loss (%)'}
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8f5e9', family='DM Sans'),
        margin=dict(l=0, r=10, t=10, b=80),
        xaxis=dict(gridcolor='rgba(82,183,136,0.1)', tickangle=-30),
        yaxis=dict(gridcolor='rgba(82,183,136,0.1)'),
        coloraxis_showscale=False
    )
    return fig

def plot_state_distribution(df):
    state_data = df.groupby('state')['yield_loss_percent'].mean().reset_index()
    fig = px.choropleth(
        state_data, locations='state', locationmode='country names',
        color='yield_loss_percent',
        color_continuous_scale=['#2d6a4f', '#f4a261', '#e63946'],
        title='Average Yield Loss by State (India)'
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8f5e9'),
        margin=dict(l=0, r=0, t=30, b=0), height=350
    )
    return fig


# ─── MAIN APP ──────────────────────────────────────────────────
def main():
    model, le, meta = load_model()
    df = load_dataset()
    features, predict_btn, crop, season = render_sidebar()

    # ── HERO BANNER
    st.markdown("""
    <div class='hero-banner'>
        <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;'>
            <div>
                <div class='hero-title'>🌾 AgroSense AI</div>
                <div class='hero-sub'>AI-Powered Crop Disease Prediction & Analytics Platform</div>
                <div style='margin-top:0.8rem; display:flex; gap:0.6rem; flex-wrap:wrap;'>
                    <span style='background:rgba(82,183,136,0.2);border:1px solid rgba(82,183,136,0.4);color:#95d5b2;padding:0.2rem 0.8rem;border-radius:20px;font-size:0.75rem;'>🤖 Random Forest AI</span>
                    <span style='background:rgba(82,183,136,0.2);border:1px solid rgba(82,183,136,0.4);color:#95d5b2;padding:0.2rem 0.8rem;border-radius:20px;font-size:0.75rem;'>📊 5,000 Samples</span>
                    <span style='background:rgba(82,183,136,0.2);border:1px solid rgba(82,183,136,0.4);color:#95d5b2;padding:0.2rem 0.8rem;border-radius:20px;font-size:0.75rem;'>🌿 10 Disease Classes</span>
                    <span style='background:rgba(82,183,136,0.2);border:1px solid rgba(82,183,136,0.4);color:#95d5b2;padding:0.2rem 0.8rem;border-radius:20px;font-size:0.75rem;'>⚡ 94.70% Accuracy</span>
                </div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:4rem;'>🛰️</div>
                <div style='font-size:0.7rem; color:rgba(149,213,178,0.6);'>Research Grade</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TOP METRICS
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Model Accuracy", "94.70%", "↑ 5-fold CV")
    with c2:
        st.metric("F1 Score", "0.9467", "Weighted avg")
    with c3:
        st.metric("Dataset Size", "5,000", "Samples")
    with c4:
        st.metric("Disease Classes", "10", "Including Healthy")
    with c5:
        st.metric("Features Used", "13", "Agronomic params")

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Disease Prediction", "📊 Data Analytics", "📈 Model Insights", "📋 Dataset Explorer"
    ])

    # ═══════════════════════════════════════════════════════════
    # TAB 1: PREDICTION
    # ═══════════════════════════════════════════════════════════
    with tab1:
        if predict_btn:
            pred_label, confidence, top3 = predict_disease(model, le, features)
            info = DISEASE_INFO.get(pred_label, DISEASE_INFO["Healthy"])

            # Determine card style
            if pred_label == "Healthy":
                card_class = "result-healthy"
            elif info["severity"] in ["Severe", "High"]:
                card_class = "result-disease"
            else:
                card_class = "result-moderate"

            st.markdown(f"""
            <div class='{card_class}'>
                <div style='font-size:3.5rem;'>{info['icon']}</div>
                <div style='font-family:Syne,sans-serif; font-size:2rem; font-weight:800; color:white; margin:0.5rem 0;'>
                    {pred_label}
                </div>
                <div style='font-size:1rem; color:rgba(255,255,255,0.75); margin-bottom:1rem;'>
                    {info['description']}
                </div>
                <div style='display:flex; justify-content:center; gap:2rem; flex-wrap:wrap;'>
                    <div>
                        <div style='font-size:0.75rem; color:rgba(255,255,255,0.5); text-transform:uppercase; letter-spacing:0.08em;'>Confidence</div>
                        <div style='font-family:Syne,sans-serif; font-size:1.8rem; color:white; font-weight:700;'>{confidence*100:.1f}%</div>
                    </div>
                    <div>
                        <div style='font-size:0.75rem; color:rgba(255,255,255,0.5); text-transform:uppercase; letter-spacing:0.08em;'>Severity</div>
                        <div style='font-family:Syne,sans-serif; font-size:1.8rem; color:white; font-weight:700;'>{info['severity']}</div>
                    </div>
                    <div>
                        <div style='font-size:0.75rem; color:rgba(255,255,255,0.5); text-transform:uppercase; letter-spacing:0.08em;'>Crop</div>
                        <div style='font-family:Syne,sans-serif; font-size:1.8rem; color:white; font-weight:700;'>{crop}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)
            col_l, col_r = st.columns([1, 1])

            with col_l:
                st.markdown("<div class='section-header'>💊 Recommended Treatment</div>", unsafe_allow_html=True)
                for t in info['treatment']:
                    st.markdown(f"<div class='treatment-box'>• {t}</div>", unsafe_allow_html=True)

                st.markdown("<div class='section-header' style='margin-top:1.2rem;'>🛡️ Prevention</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='treatment-box'>{info['prevention']}</div>", unsafe_allow_html=True)

            with col_r:
                st.markdown("<div class='section-header'>📊 Top 3 Predictions</div>", unsafe_allow_html=True)
                for disease, prob in top3:
                    d_info = DISEASE_INFO.get(disease, {})
                    icon = d_info.get('icon', '🌿')
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"**{icon} {disease}**")
                        st.progress(prob)
                    with col_b:
                        st.markdown(f"<div style='color:white;font-size:1.1rem;font-weight:700;margin-top:0.3rem;'>{prob*100:.1f}%</div>", unsafe_allow_html=True)
                    st.markdown("<br/>", unsafe_allow_html=True)

                # Input summary
                st.markdown("<div class='section-header' style='margin-top:1rem;'>📝 Input Summary</div>", unsafe_allow_html=True)
                summary_df = pd.DataFrame({
                    'Parameter': ['Temperature', 'Humidity', 'Rainfall', 'Soil pH', 'Nitrogen', 'Phosphorus', 'Potassium'],
                    'Value': [f"{features['temperature_celsius']}°C",
                              f"{features['humidity_percent']}%",
                              f"{features['rainfall_mm']} mm",
                              str(features['soil_ph']),
                              f"{features['nitrogen_kg_ha']} kg/ha",
                              f"{features['phosphorus_kg_ha']} kg/ha",
                              f"{features['potassium_kg_ha']} kg/ha"]
                })
                st.dataframe(summary_df, hide_index=True, use_container_width=True)

        else:
            st.markdown("""
            <div style='text-align:center; padding:4rem 2rem; background:rgba(255,255,255,0.04);
                 border:2px dashed rgba(82,183,136,0.3); border-radius:24px;'>
                <div style='font-size:4rem;'>🌱</div>
                <div style='font-family:Syne,sans-serif; font-size:1.5rem; font-weight:700; color:#52b788; margin:1rem 0;'>
                    Ready to Diagnose Your Crop
                </div>
                <div style='color:rgba(255,255,255,0.5); max-width:400px; margin:0 auto; font-size:0.9rem;'>
                    Enter your field parameters in the sidebar on the left, then click <strong style='color:#52b788;'>Predict Disease</strong> to get AI-powered diagnosis with treatment recommendations.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # TAB 2: DATA ANALYTICS
    # ═══════════════════════════════════════════════════════════
    with tab2:
        st.markdown("<div class='section-header'>📊 Dataset Overview & Distribution</div>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Records", f"{len(df):,}")
        with m2:
            st.metric("Healthy Crops", f"{(df['disease_label']=='Healthy').sum():,}")
        with m3:
            st.metric("Diseased Crops", f"{(df['disease_label']!='Healthy').sum():,}")
        with m4:
            st.metric("Avg Yield Loss", f"{df[df['disease_label']!='Healthy']['yield_loss_percent'].mean():.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-header'>Disease Distribution</div>", unsafe_allow_html=True)
            st.plotly_chart(plot_disease_distribution(df), use_container_width=True)

        with col2:
            st.markdown("<div class='section-header'>Average Yield Loss by Disease</div>", unsafe_allow_html=True)
            st.plotly_chart(plot_yield_loss(df), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("<div class='section-header'>Temperature vs Humidity (by Disease)</div>", unsafe_allow_html=True)
            st.plotly_chart(plot_scatter(df, 'temperature_celsius', 'humidity_percent'), use_container_width=True)

        with col4:
            st.markdown("<div class='section-header'>Rainfall vs Soil pH (by Disease)</div>", unsafe_allow_html=True)
            st.plotly_chart(plot_scatter(df, 'rainfall_mm', 'soil_ph'), use_container_width=True)

        st.markdown("<div class='section-header'>Feature Correlation Matrix</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_correlation(df), use_container_width=True)

    # ═══════════════════════════════════════════════════════════
    # TAB 3: MODEL INSIGHTS
    # ═══════════════════════════════════════════════════════════
    with tab3:
        st.markdown("<div class='section-header'>🤖 Model Performance & Insights</div>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Accuracy", f"{meta['accuracy']*100:.2f}%")
        with m2:
            st.metric("F1 Score", f"{meta['f1_score']:.4f}")
        with m3:
            st.metric("Precision", f"{meta['precision']:.4f}")
        with m4:
            st.metric("Recall", f"{meta['recall']:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-header'>Feature Importance</div>", unsafe_allow_html=True)
            st.plotly_chart(plot_feature_importance(meta), use_container_width=True)

        with col2:
            st.markdown("<div class='section-header'>Cross-Validation Scores</div>", unsafe_allow_html=True)
            cv_fig = go.Figure()
            cv_fig.add_trace(go.Bar(
                x=[f"Fold {i+1}" for i in range(5)],
                y=meta['cv_scores'],
                marker_color=['#52b788', '#95d5b2', '#52b788', '#95d5b2', '#52b788'],
                text=[f"{v*100:.2f}%" for v in meta['cv_scores']],
                textposition='auto'
            ))
            cv_fig.add_hline(y=meta['cv_mean'], line_dash="dash", line_color="#f4a261",
                             annotation_text=f"Mean: {meta['cv_mean']*100:.2f}%")
            cv_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e8f5e9', family='DM Sans'),
                margin=dict(l=0, r=10, t=10, b=0),
                yaxis=dict(gridcolor='rgba(82,183,136,0.1)', tickformat='.1%'),
                xaxis=dict(gridcolor='rgba(0,0,0,0)')
            )
            st.plotly_chart(cv_fig, use_container_width=True)

        # Per-class performance
        st.markdown("<div class='section-header'>Per-Class Classification Report</div>", unsafe_allow_html=True)
        if 'classification_report' in meta:
            cr = meta['classification_report']
            rows = []
            for cls in meta['classes']:
                if cls in cr:
                    rows.append({
                        'Disease': cls,
                        'Precision': f"{cr[cls]['precision']:.3f}",
                        'Recall': f"{cr[cls]['recall']:.3f}",
                        'F1-Score': f"{cr[cls]['f1-score']:.3f}",
                        'Support': int(cr[cls]['support'])
                    })
            cr_df = pd.DataFrame(rows)
            st.dataframe(cr_df, hide_index=True, use_container_width=True)

        # Model architecture info
        st.markdown("<div class='section-header'>Model Architecture</div>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            arch_info = {
                "Algorithm": "Random Forest Classifier",
                "N Estimators": "200 decision trees",
                "Max Depth": "20 levels",
                "Min Samples Split": "5",
                "Max Features": "sqrt(n_features)",
                "Preprocessing": "StandardScaler + Imputer"
            }
            for k, v in arch_info.items():
                st.markdown(f"<div class='treatment-box'><strong>{k}:</strong> {v}</div>", unsafe_allow_html=True)

        with col_b:
            train_info = {
                "Training Samples": f"{meta['train_samples']:,}",
                "Test Samples": f"{meta['test_samples']:,}",
                "Train/Test Split": "80% / 20%",
                "Stratification": "Yes (class-balanced)",
                "CV Folds": "5-fold cross-validation",
                "CV Mean±Std": f"{meta['cv_mean']*100:.2f}% ± {meta['cv_std']*100:.2f}%"
            }
            for k, v in train_info.items():
                st.markdown(f"<div class='treatment-box'><strong>{k}:</strong> {v}</div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # TAB 4: DATASET EXPLORER
    # ═══════════════════════════════════════════════════════════
    with tab4:
        st.markdown("<div class='section-header'>📋 Dataset Explorer</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            filter_disease = st.multiselect(
                "Filter by Disease", options=sorted(df['disease_label'].unique()),
                default=list(df['disease_label'].unique()[:3])
            )
        with col2:
            filter_crop = st.multiselect(
                "Filter by Crop", options=sorted(df['crop_type'].unique()),
                default=list(df['crop_type'].unique()[:3])
            )
        with col3:
            n_rows = st.selectbox("Rows to display", [25, 50, 100, 250, 500], index=1)

        filtered = df[
            (df['disease_label'].isin(filter_disease if filter_disease else df['disease_label'].unique())) &
            (df['crop_type'].isin(filter_crop if filter_crop else df['crop_type'].unique()))
        ].head(n_rows)

        st.markdown(f"<div style='color:#95d5b2;font-size:0.85rem;margin-bottom:0.8rem;'>Showing {len(filtered)} of {len(df)} records</div>", unsafe_allow_html=True)

        display_cols = ['sample_id', 'crop_type', 'state', 'season', 'temperature_celsius',
                        'humidity_percent', 'rainfall_mm', 'soil_ph', 'disease_label',
                        'severity_label', 'yield_loss_percent', 'treatment_cost_inr']
        st.dataframe(filtered[display_cols], hide_index=True, use_container_width=True)

        # Download
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Full Dataset (CSV)",
            data=csv_data,
            file_name="crop_disease_dataset.csv",
            mime="text/csv"
        )

    # Footer
    st.markdown("""
    <div style='text-align:center; padding:2rem 0 1rem 0; margin-top:2rem;
         border-top:1px solid rgba(82,183,136,0.15);'>
        <div style='color:rgba(149,213,178,0.5); font-size:0.75rem;'>
            🌾 AgroSense AI | Crop Disease Prediction System | Built with Streamlit + Scikit-Learn<br/>
            Research-Grade Application | Indian Agriculture | Dataset: 5,000 samples | 10 Disease Classes
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
