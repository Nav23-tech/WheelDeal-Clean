"""
FairRide AI — Explainable Used Car Price & Trust Estimator
Complete implementation with price prediction, SHAP explainability, and trust scoring

SETUP INSTRUCTIONS:
1. Save this file as fairride_app.py
2. Open Terminal in VS Code (Ctrl + `)
3. Run: python3 -m venv venv
4. Run: source venv/bin/activate
5. Run: pip install streamlit pandas numpy scikit-learn xgboost shap matplotlib seaborn
6. Run: streamlit run fairride_app.py
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
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="FairRide AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .trust-high {
        color: #28a745;
        font-weight: bold;
    }
    .trust-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .trust-low {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


class FairRideAI:
    """Main class for FairRide AI system"""
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_names = []
        self.anomaly_detector = None
        self.explainer = None
        
    def generate_sample_data(self, n_samples=1000):
        """Generate realistic sample used car data"""
        np.random.seed(42)
        
        brands = ['Maruti', 'Hyundai', 'Honda', 'Tata', 'Mahindra', 'Toyota', 'Ford', 'Volkswagen']
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
        
        # Generate realistic price based on features
        base_price = 500000
        df['price'] = (
            base_price * 
            (1 - (2024 - df['year']) * 0.08) *  # Depreciation
            (1 - df['km_driven'] / 1000000) *  # Mileage effect
            (df['engine_cc'] / 1500) *  # Engine size
            np.where(df['transmission'] == 'Automatic', 1.15, 1.0) *  # Auto premium
            np.where(df['fuel_type'] == 'Diesel', 1.1, 1.0) *  # Diesel premium
            np.where(df['owner_type'] == 'First', 1.2, 0.9)  # Owner effect
        )
        
        # Add some noise and ensure positive prices
        df['price'] = df['price'] * np.random.uniform(0.8, 1.2, n_samples)
        df['price'] = df['price'].clip(lower=100000)
        df['price'] = df['price'].round(-3)  # Round to nearest thousand
        
        # Add some anomalies (5%)
        anomaly_idx = np.random.choice(df.index, size=int(0.05 * n_samples), replace=False)
        df.loc[anomaly_idx, 'price'] *= np.random.choice([0.5, 1.8], len(anomaly_idx))
        
        return df
    
    def preprocess_data(self, df, fit=True):
        """Preprocess the data for training/prediction"""
        df_processed = df.copy()
        
        # Calculate car age
        current_year = datetime.now().year
        df_processed['car_age'] = current_year - df_processed['year']
        
        # Encode categorical variables
        categorical_cols = ['brand', 'fuel_type', 'transmission', 'owner_type']
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col + '_encoded'] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                # Handle unseen categories
                df_processed[col + '_encoded'] = df_processed[col].map(
                    lambda x: self.label_encoders[col].transform([x])[0] 
                    if x in self.label_encoders[col].classes_ 
                    else -1
                )
        
        # Select features for modeling
        feature_cols = [
            'car_age', 'km_driven', 'engine_cc', 'mileage', 'seats',
            'brand_encoded', 'fuel_type_encoded', 'transmission_encoded', 'owner_type_encoded'
        ]
        
        if fit:
            self.feature_names = feature_cols
        
        return df_processed[feature_cols], df_processed
    
    def train_model(self, df):
        """Train the XGBoost model and anomaly detector"""
        st.info("🔄 Training model on sample data...")
        
        # Prepare data
        X, df_processed = self.preprocess_data(df, fit=True)
        y = df['price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            objective='reg:squarederror'
        )
        self.model.fit(X_train, y_train)
        
        # Train anomaly detector
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.anomaly_detector.fit(X_train)
        
        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        # Calculate metrics
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        st.success(f"✅ Model trained! R² Score: {test_score:.3f}")
        
        return train_score, test_score
    
    def predict_price(self, input_data):
        """Predict price for input data"""
        X, _ = self.preprocess_data(input_data, fit=False)
        prediction = self.model.predict(X)[0]
        return prediction
    
    def get_shap_explanation(self, input_data):
        """Get SHAP values and top features"""
        X, _ = self.preprocess_data(input_data, fit=False)
        shap_values = self.explainer.shap_values(X)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_values[0]
        })
        feature_importance['abs_shap'] = abs(feature_importance['shap_value'])
        feature_importance = feature_importance.sort_values('abs_shap', ascending=False)
        
        return feature_importance, shap_values
    
    def calculate_trust_score(self, input_data, predicted_price):
        """Calculate trust score based on anomaly detection and consistency checks"""
        X, df_processed = self.preprocess_data(input_data, fit=False)
        
        scores = []
        explanations = []
        
        # 1. Anomaly Detection Score (40 points)
        anomaly_score = self.anomaly_detector.score_samples(X)[0]
        # Normalize to 0-40
        anomaly_points = min(40, max(0, (anomaly_score + 0.5) * 40))
        scores.append(anomaly_points)
        explanations.append(f"Anomaly Check: {anomaly_points:.1f}/40")
        
        # 2. Price-to-Age Consistency (20 points)
        car_age = df_processed['car_age'].values[0]
        expected_depreciation = 0.92 ** car_age
        age_consistency = min(20, 20 * expected_depreciation)
        scores.append(age_consistency)
        explanations.append(f"Age Consistency: {age_consistency:.1f}/20")
        
        # 3. Mileage Consistency (20 points)
        km_driven = input_data['km_driven'].values[0]
        expected_km = car_age * 15000
        km_ratio = min(km_driven / expected_km, expected_km / km_driven) if expected_km > 0 else 1
        mileage_points = 20 * km_ratio
        scores.append(mileage_points)
        explanations.append(f"Mileage Consistency: {mileage_points:.1f}/20")
        
        # 4. Price Range Check (20 points)
        # Check if price is within reasonable bounds
        if predicted_price > 50000 and predicted_price < 5000000:
            price_points = 20
        elif predicted_price > 30000 and predicted_price < 10000000:
            price_points = 15
        else:
            price_points = 10
        scores.append(price_points)
        explanations.append(f"Price Range Check: {price_points:.1f}/20")
        
        total_score = sum(scores)
        
        return total_score, explanations
    
    def get_trust_category(self, score):
        """Categorize trust score"""
        if score >= 75:
            return "High Trust", "trust-high", "🟢"
        elif score >= 50:
            return "Medium Trust", "trust-medium", "🟡"
        else:
            return "Low Trust", "trust-low", "🔴"


# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = FairRideAI()
    st.session_state.trained = False

model = st.session_state.model

# Header
st.markdown('<p class="main-header">🚗 FairRide AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explainable Used Car Price & Trust Estimator</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Model Management")
    
    if st.button("🔄 Train Model", type="primary", use_container_width=True):
        sample_data = model.generate_sample_data(n_samples=1000)
        train_score, test_score = model.train_model(sample_data)
        st.session_state.trained = True
        st.session_state.sample_data = sample_data
    
    if st.session_state.trained:
        st.success("✅ Model Ready")
        st.metric("Training Samples", "1,000")
    else:
        st.warning("⚠️ Please train model first")
    
    st.divider()
    st.markdown("### 🎯 About")
    st.markdown("""
    **FairRide AI** provides:
    - 🔮 ML-based price prediction
    - 📊 SHAP explainability
    - 🛡️ Trust score (0-100)
    - 🎯 Transparent insights
    """)

# Main content
if not st.session_state.trained:
    st.info("👈 Please train the model using the sidebar button to get started!")
    st.stop()

# Input form
st.header("🚗 Vehicle Details")

col1, col2, col3 = st.columns(3)

with col1:
    brand = st.selectbox("Brand", ['Maruti', 'Hyundai', 'Honda', 'Tata', 'Mahindra', 'Toyota', 'Ford', 'Volkswagen'])
    year = st.number_input("Year of Manufacture", min_value=2000, max_value=2024, value=2018)
    fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'Electric'])

with col2:
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=5000)
    transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
    owner_type = st.selectbox("Owner Type", ['First', 'Second', 'Third', 'Fourth & Above'])

with col3:
    engine_cc = st.number_input("Engine CC", min_value=500, max_value=5000, value=1500, step=100)
    mileage = st.number_input("Mileage (km/l)", min_value=5.0, max_value=40.0, value=18.0, step=0.5)
    seats = st.selectbox("Number of Seats", [5, 7, 8])

# Predict button
if st.button("🔍 Estimate Price & Trust Score", type="primary", use_container_width=True):
    # Create input dataframe
    input_data = pd.DataFrame({
        'brand': [brand],
        'year': [year],
        'km_driven': [km_driven],
        'fuel_type': [fuel_type],
        'transmission': [transmission],
        'owner_type': [owner_type],
        'engine_cc': [engine_cc],
        'mileage': [mileage],
        'seats': [seats]
    })
    
    # Get predictions
    predicted_price = model.predict_price(input_data)
    trust_score, trust_explanations = model.calculate_trust_score(input_data, predicted_price)
    trust_category, trust_class, trust_emoji = model.get_trust_category(trust_score)
    feature_importance, shap_values = model.get_shap_explanation(input_data)
    
    # Display results
    st.divider()
    st.header("📈 Results")
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="💰 Estimated Fair Price",
            value=f"₹{predicted_price:,.0f}",
            delta=f"±₹{predicted_price * 0.05:,.0f}"
        )
    
    with col2:
        st.metric(
            label=f"{trust_emoji} Trust Score",
            value=f"{trust_score:.1f}/100"
        )
        st.markdown(f'<p class="{trust_class}">{trust_category}</p>', unsafe_allow_html=True)
    
    with col3:
        price_per_year = predicted_price / (2024 - year)
        st.metric(
            label="📊 Value per Year",
            value=f"₹{price_per_year:,.0f}"
        )
    
    # SHAP Explanation
    st.divider()
    st.subheader("🔍 Top 5 Factors Affecting Price")
    
    top_5 = feature_importance.head(5)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#28a745' if x > 0 else '#dc3545' for x in top_5['shap_value']]
    ax.barh(range(len(top_5)), top_5['shap_value'], color=colors)
    ax.set_yticks(range(len(top_5)))
    ax.set_yticklabels(top_5['feature'])
    ax.set_xlabel('SHAP Value (Impact on Price)')
    ax.set_title('Feature Importance - What Drives Your Car\'s Price?')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Feature interpretation
    st.markdown("**Interpretation:**")
    for idx, row in top_5.iterrows():
        impact = "increases" if row['shap_value'] > 0 else "decreases"
        st.markdown(f"- **{row['feature']}**: {impact} price by ₹{abs(row['shap_value']):,.0f}")
    
    # Trust Score Breakdown
    st.divider()
    st.subheader("🛡️ Trust Score Breakdown")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        for explanation in trust_explanations:
            st.markdown(f"- {explanation}")
        
        st.markdown("""
        **Trust Score Guide:**
        - 🟢 **75-100**: Highly trustworthy listing
        - 🟡 **50-74**: Moderately trustworthy, verify details
        - 🔴 **0-49**: Low trust, investigate thoroughly
        """)
    
    with col2:
        # Trust score gauge
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(['Trust Score'], [trust_score], color='#1f77b4', height=0.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Score')
        ax.set_title(f'Trust Score: {trust_score:.1f}/100')
        
        # Add color zones
        ax.axvspan(0, 50, alpha=0.2, color='red')
        ax.axvspan(50, 75, alpha=0.2, color='yellow')
        ax.axvspan(75, 100, alpha=0.2, color='green')
        
        plt.tight_layout()
        st.pyplot(fig)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>FairRide AI</strong> — Making used car valuation transparent and trustworthy</p>
    <p>Built with Python, XGBoost, SHAP & Streamlit | 100% Local & Private</p>
</div>
""", unsafe_allow_html=True)