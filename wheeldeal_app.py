"""
WheelDeal AI — Smart Car Price & Trust Estimator
Your trusted partner for transparent used car valuation
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="WheelDeal AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
        padding: 2rem;
    }
    
    .stApp {
        background: transparent;
    }
    
    .hero-title {
        font-size: 4.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease;
        text-shadow: 0 0 30px rgba(96, 165, 250, 0.5);
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        color: #e0e7ff;
        text-align: center;
        margin-bottom: 3rem;
        animation: fadeInUp 0.8s ease;
    }
    
    .price-card {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 20px 40px rgba(37, 99, 235, 0.4);
        animation: float 3s ease-in-out infinite;
        border: 2px solid rgba(255,255,255,0.1);
    }
    
    .price-value {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 1rem 0;
        text-shadow: 0 0 20px rgba(255,255,255,0.5);
    }
    
    .trust-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border-left: 6px solid;
        transition: all 0.3s ease;
    }
    
    .trust-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 50px rgba(0,0,0,0.3);
    }
    
    .trust-high {
        border-color: #10b981;
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    }
    
    .trust-medium {
        border-color: #f59e0b;
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    }
    
    .trust-low {
        border-color: #ef4444;
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    }
    
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 30px;
        margin: 0.5rem;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        color: white;
        border: none;
        padding: 1.2rem 2.5rem;
        border-radius: 15px;
        font-size: 1.2rem;
        font-weight: 700;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(37, 99, 235, 0.6);
        background: linear-gradient(135deg, #1d4ed8 0%, #6d28d9 100%);
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    
    .metric-container {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid rgba(37, 99, 235, 0.1);
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        border-color: rgba(37, 99, 235, 0.3);
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .info-box {
        background: rgba(255,255,255,0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)


class WheelDealAI:
    """Main class for WheelDeal AI system"""
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_names = []
        self.anomaly_detector = None
        self.explainer = None
        
    def generate_sample_data(self, n_samples=1000):
        """Generate realistic sample used car data"""
        np.random.seed(42)
        
        brands = ['Maruti', 'Hyundai', 'Honda', 'Tata', 'Mahindra', 'Toyota', 'Ford', 'Volkswagen', 'Kia', 'Renault']
        fuel_types = ['Petrol', 'Diesel', 'CNG', 'Electric']
        transmission = ['Manual', 'Automatic']
        owner_types = ['First', 'Second', 'Third', 'Fourth & Above']
        
        data = {
            'brand': np.random.choice(brands, n_samples),
            'year': np.random.randint(2010, 2024, n_samples),
            'km_driven': np.random.randint(5000, 150000, n_samples),
            'fuel_type': np.random.choice(fuel_types, n_samples, p=[0.5, 0.35, 0.1, 0.05]),
            'transmission': np.random.choice(transmission, n_samples, p=[0.7, 0.3]),
            'owner_type': np.random.choice(owner_types, n_samples, p=[0.5, 0.3, 0.15, 0.05]),
            'engine_cc': np.random.randint(800, 2500, n_samples),
            'mileage': np.random.uniform(10, 30, n_samples),
            'seats': np.random.choice([5, 7, 8], n_samples, p=[0.8, 0.15, 0.05])
        }
        
        df = pd.DataFrame(data)
        
        # Generate realistic price
        base_price = 500000
        df['price'] = (
            base_price * 
            (1 - (2024 - df['year']) * 0.08) *
            (1 - df['km_driven'] / 1000000) *
            (df['engine_cc'] / 1500) *
            np.where(df['transmission'] == 'Automatic', 1.15, 1.0) *
            np.where(df['fuel_type'] == 'Diesel', 1.1, 1.0) *
            np.where(df['owner_type'] == 'First', 1.2, 0.9)
        )
        
        df['price'] = df['price'] * np.random.uniform(0.8, 1.2, n_samples)
        df['price'] = df['price'].clip(lower=100000)
        df['price'] = df['price'].round(-3)
        
        anomaly_idx = np.random.choice(df.index, size=int(0.05 * n_samples), replace=False)
        df.loc[anomaly_idx, 'price'] *= np.random.choice([0.5, 1.8], len(anomaly_idx))
        
        return df
    
    def preprocess_data(self, df, fit=True):
        """Preprocess data"""
        df_processed = df.copy()
        current_year = datetime.now().year
        df_processed['car_age'] = current_year - df_processed['year']
        
        categorical_cols = ['brand', 'fuel_type', 'transmission', 'owner_type']
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col + '_encoded'] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                # map unseen labels to -1
                le = self.label_encoders.get(col, None)
                if le is None:
                    df_processed[col + '_encoded'] = -1
                else:
                    mapped = []
                    for v in df_processed[col].values:
                        if v in list(le.classes_):
                            mapped.append(int(le.transform([v])[0]))
                        else:
                            mapped.append(-1)
                    df_processed[col + '_encoded'] = mapped
        
        feature_cols = [
            'car_age', 'km_driven', 'engine_cc', 'mileage', 'seats',
            'brand_encoded', 'fuel_type_encoded', 'transmission_encoded', 'owner_type_encoded'
        ]

        if fit:
            self.feature_names = feature_cols
        
        return df_processed[feature_cols], df_processed
    
    def train_model(self, df):
        """Train model"""
        # Preprocess & build training set
        X, df_processed = self.preprocess_data(df, fit=True)
        # y must be taken from original df (price present)
        y = df['price']
        
        # Ensure numeric
        X = X.apply(pd.to_numeric, errors="coerce")
        y = pd.to_numeric(y, errors="coerce")
        X = X.fillna(0)
        y = y.fillna(0)
        
        # Train split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            objective='reg:squarederror'
        )
        self.model.fit(X_train, y_train)
        
        # SHAP explainer (safe initialization)
        try:
            self.explainer = shap.Explainer(self.model, X_train)
        except Exception:
            # fallback
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception:
                self.explainer = None
        
        # Anomaly detector
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_detector.fit(X_train)
        
        # Evaluate
        test_score = self.model.score(X_test, y_test)
        return test_score
    
    def predict_price(self, input_data):
        """Predict price for input_data (DataFrame with same raw columns)"""
        X, _ = self.preprocess_data(input_data, fit=False)
        # ensure numeric & same ordering
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        pred = self.model.predict(X)
        # return scalar
        return float(pred[0])
    
    def get_shap_explanation(self, input_data):
        """Get SHAP values and feature importance for an input row"""
        X, _ = self.preprocess_data(input_data, fit=False)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        
        if self.explainer is None:
            # no explainer available
            shap_values = np.zeros((1, len(self.feature_names)))
        else:
            try:
                expl = self.explainer(X)
                # SHAP library returns different types depending on version
                if hasattr(expl, "values"):
                    shap_values = np.array(expl.values)
                else:
                    shap_values = np.array(expl)
            except Exception:
                # fallback: try shap.TreeExplainer
                try:
                    te = shap.TreeExplainer(self.model)
                    shap_values = np.array(te.shap_values(X))
                except Exception:
                    shap_values = np.zeros((1, len(self.feature_names)))
        
        # Build feature importance dataframe
        # Ensure shap_values has shape (1, n_features)
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_values[0][:len(self.feature_names)]
        })
        feature_importance['abs_shap'] = feature_importance['shap_value'].abs()
        feature_importance = feature_importance.sort_values('abs_shap', ascending=False)
        
        return feature_importance, shap_values
    
    def calculate_trust_score(self, input_data, predicted_price):
        """Calculate trust score"""
        X, df_processed = self.preprocess_data(input_data, fit=False)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        scores = []
        explanations = []
        
        # Anomaly sample score (higher = more normal for IsolationForest.score_samples)
        if self.anomaly_detector is not None:
            try:
                anomaly_score = self.anomaly_detector.score_samples(X)[0]
            except Exception:
                anomaly_score = 0.0
        else:
            anomaly_score = 0.0
        anomaly_points = min(40, max(0, (anomaly_score + 0.5) * 40))
        scores.append(anomaly_points)
        explanations.append(f"Pattern Analysis: {anomaly_points:.1f}/40")
        
        # Age consistency
        car_age = int(df_processed['car_age'].values[0])
        expected_depreciation = 0.92 ** car_age
        age_consistency = min(20, 20 * expected_depreciation)
        scores.append(age_consistency)
        explanations.append(f"Age Consistency: {age_consistency:.1f}/20")
        
        # Mileage check
        km_driven = float(input_data['km_driven'].values[0])
        expected_km = car_age * 15000 if car_age > 0 else 15000
        km_ratio = min(km_driven / expected_km, expected_km / km_driven) if expected_km > 0 and km_driven > 0 else 1
        mileage_points = 20 * km_ratio
        scores.append(mileage_points)
        explanations.append(f"Mileage Check: {mileage_points:.1f}/20")
        
        # Price validity
        if 50000 < predicted_price < 5000000:
            price_points = 20
        elif 30000 < predicted_price < 10000000:
            price_points = 15
        else:
            price_points = 10
        scores.append(price_points)
        explanations.append(f"Price Validity: {price_points:.1f}/20")
        
        total = sum(scores)
        # Normalize to 0-100 if rounding issues
        total = max(0, min(100, total))
        return total, explanations
    
    def get_trust_category(self, score):
        """Categorize trust score"""
        if score >= 75:
            return "High Trust", "trust-high", "🟢", "#10b981"
        elif score >= 50:
            return "Medium Trust", "trust-medium", "🟡", "#f59e0b"
        else:
            return "Low Trust", "trust-low", "🔴", "#ef4444"


# Initialize
if 'model' not in st.session_state:
    st.session_state.model = WheelDealAI()
    st.session_state.trained = False

model = st.session_state.model

# Header
st.markdown('<p class="hero-title">🚗 WheelDeal AI</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Your Trusted Partner for Smart Car Valuation</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Control Panel")
    
    if st.button("🚀 Train AI Model", use_container_width=True):
        with st.spinner("Training AI on 1000 cars..."):
            sample_data = model.generate_sample_data(n_samples=1000)
            test_score = model.train_model(sample_data)
            st.session_state.trained = True
            st.balloons()
            st.success(f"✅ AI Ready!")
            # show percentage
            try:
                st.metric("Accuracy", f"{test_score*100:.1f}%")
            except Exception:
                st.metric("Accuracy", "N/A")
    
    if st.session_state.trained:
        st.success("🤖 AI Active")
    else:
        st.warning("⚠️ Train AI first")
    
    st.divider()
    
    st.markdown("### 💎 Features")
    st.markdown('<span class="feature-badge">🤖 XGBoost ML</span>', unsafe_allow_html=True)
    st.markdown('<span class="feature-badge">🔍 Explainable AI</span>', unsafe_allow_html=True)
    st.markdown('<span class="feature-badge">🛡️ Trust Score</span>', unsafe_allow_html=True)
    st.markdown('<span class="feature-badge">📊 Real-time</span>', unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### ⚡ Powered By")
    st.markdown("**Advanced ML & AI**")
    st.markdown("*XGBoost • SHAP • Python*")

# Main content
if not st.session_state.trained:
    st.markdown("""
        <div class="info-box">
            <h2 style="text-align: center; color: #2563eb;">👋 Welcome to WheelDeal AI!</h2>
            <p style="text-align: center; font-size: 1.2rem; color: #666;">
                Get instant, transparent car valuations powered by Machine Learning
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎯 Accurate Prices")
        st.write("ML-powered predictions using XGBoost on 1000+ cars")
    with col2:
        st.markdown("### 🔍 Full Transparency")
        st.write("See exactly what affects your car's value")
    with col3:
        st.markdown("### 🛡️ Trust Verified")
        st.write("Detect suspicious deals with AI scoring")
    
    st.info("👈 **Start by training the AI model in the sidebar!**")
    st.stop()

# Input form
st.markdown("## 🚗 Enter Your Car Details")

col1, col2, col3 = st.columns(3)

with col1:
    brand = st.selectbox("🏢 Brand", ['Maruti', 'Hyundai', 'Honda', 'Tata', 'Mahindra', 'Toyota', 'Ford', 'Volkswagen', 'Kia', 'Renault'])
    year = st.number_input("📅 Year", min_value=2000, max_value=2024, value=2018)
    fuel_type = st.selectbox("⛽ Fuel", ['Petrol', 'Diesel', 'CNG', 'Electric'])

with col2:
    km_driven = st.number_input("🛣️ Kilometers", min_value=0, max_value=500000, value=50000, step=5000)
    transmission = st.selectbox("⚙️ Transmission", ['Manual', 'Automatic'])
    owner_type = st.selectbox("👤 Owner", ['First', 'Second', 'Third', 'Fourth & Above'])

with col3:
    engine_cc = st.number_input("🔧 Engine CC", min_value=500, max_value=5000, value=1500, step=100)
    mileage = st.number_input("⛽ Mileage (km/l)", min_value=5.0, max_value=40.0, value=18.0, step=0.5)
    seats = st.selectbox("💺 Seats", [5, 7, 8])

st.markdown("")
if st.button("🔮 GET FAIR PRICE & TRUST SCORE", type="primary", use_container_width=True):
    with st.spinner("🤖 AI analyzing your car..."):
        input_data = pd.DataFrame({
            'brand': [brand], 'year': [year], 'km_driven': [km_driven],
            'fuel_type': [fuel_type], 'transmission': [transmission],
            'owner_type': [owner_type], 'engine_cc': [engine_cc],
            'mileage': [mileage], 'seats': [seats]
        })
        
        predicted_price = model.predict_price(input_data)
        trust_score, trust_explanations = model.calculate_trust_score(input_data, predicted_price)
        trust_category, trust_class, trust_emoji, trust_color = model.get_trust_category(trust_score)
        feature_importance, shap_values = model.get_shap_explanation(input_data)
        
        st.markdown("---")
        st.markdown("## 🎯 Your Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class="price-card">
                    <div style="font-size: 1.3rem; opacity: 0.9;">💰 Fair Market Price</div>
                    <div class="price-value">₹{predicted_price:,.0f}</div>
                    <div style="font-size: 1rem; opacity: 0.8;">±₹{predicted_price * 0.05:,.0f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="trust-card {trust_class}">
                    <div style="font-size: 1.2rem; color: #666; margin-bottom: 0.5rem;">{trust_emoji} Trust Score</div>
                    <div style="font-size: 3.5rem; font-weight: 700; color: {trust_color};">{trust_score:.0f}</div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: {trust_color}; margin-top: 0.5rem;">{trust_category}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            car_age = 2024 - year
            value_per_year = predicted_price / car_age if car_age > 0 else predicted_price
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">📈 Value Per Year</div>
                    <div class="metric-value">₹{value_per_year:,.0f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## 🔍 What Affects Your Price?")
        
        top_5 = feature_importance.head(5)
        colors = ['#10b981' if x > 0 else '#ef4444' for x in top_5['shap_value']]
        
        fig = go.Figure(go.Bar(
            x=top_5['shap_value'],
            y=top_5['feature'],
            orientation='h',
            marker=dict(color=colors, line=dict(color='white', width=2)),
            text=[f"₹{abs(x):,.0f}" for x in top_5['shap_value']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Top 5 Price Factors",
            xaxis_title="Impact on Price (₹)",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Poppins', size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📈 Increases Price")
            for idx, row in top_5[top_5['shap_value'] > 0].iterrows():
                st.markdown(f"✓ **{row['feature']}**: +₹{abs(row['shap_value']):,.0f}")
        
        with col2:
            st.markdown("### 📉 Decreases Price")
            for idx, row in top_5[top_5['shap_value'] < 0].iterrows():
                st.markdown(f"✗ **{row['feature']}**: -₹{abs(row['shap_value']):,.0f}")
        
        st.markdown("---")
        st.markdown("## 🛡️ Trust Score Details")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            for exp in trust_explanations:
                st.markdown(f"✓ {exp}")
            
            st.markdown(f"""
            **Score Guide:**
            - 🟢 **75-100**: Excellent deal - Highly trustworthy
            - 🟡 **50-74**: Good deal - Verify details
            - 🔴 **0-49**: Risky - Investigate carefully
            """)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=trust_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Trust Level"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': trust_color},
                    'steps': [
                        {'range': [0, 50], 'color': "#fee2e2"},
                        {'range': [50, 75], 'color': "#fef3c7"},
                        {'range': [75, 100], 'color': "#d1fae5"}
                    ]
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #e0e7ff; padding: 2rem;'>
        <h2 style='background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            🚗 WheelDeal AI
        </h2>
        <p style='font-size: 1.2rem;'><strong>Smart Car Valuation Made Simple</strong></p>
        <p>Powered by XGBoost ML • SHAP Explainability • Anomaly Detection</p>
        <p style='margin-top: 1rem;'>Bringing transparency to used car pricing</p>
    </div>
""", unsafe_allow_html=True)
