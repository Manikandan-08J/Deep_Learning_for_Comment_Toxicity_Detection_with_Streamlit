"""
Content Monetization Modeler — Streamlit App (Fixed)
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Revenue Predictor",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #FF0000; }
    .prediction-box {
        background: linear-gradient(135deg, #FF0000 0%, #cc0000 100%);
        color: white; padding: 1.5rem; border-radius: 12px;
        text-align: center; font-size: 2rem; font-weight: bold; margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa; border-left: 4px solid #FF0000;
        padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📺 YouTube Ad Revenue Predictor</div>', unsafe_allow_html=True)
st.markdown("**Content Monetization Modeler — ML-Powered Prediction & Analytics**")
st.markdown("---")

# ─── Find Dataset ────────────────────────────────────────────────────────────
def find_file(filename):
    search_paths = [
        filename,
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join('data', filename),
    ]
    for path in search_paths:
        if os.path.exists(path):
            return path
    return None

# ─── Load Dataset ────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    for col in ['likes', 'comments', 'watch_time_minutes']:
        df[col] = df[col].fillna(df[col].median())
    df['engagement_rate'] = (df['likes'] + df['comments']) / df['views']
    df['avg_watch_time']  = df['watch_time_minutes'] / df['views']
    df['like_ratio']      = df['likes'] / df['views']
    return df

# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_paths  = ['model/best_model.pkl',       'best_model.pkl']
    scaler_paths = ['model/scaler.pkl',            'scaler.pkl']
    col_paths    = ['model/feature_columns.pkl',   'feature_columns.pkl']

    model, scaler, columns = None, None, None
    for p in model_paths:
        if os.path.exists(p): model = joblib.load(p); break
    for p in scaler_paths:
        if os.path.exists(p): scaler = joblib.load(p); break
    for p in col_paths:
        if os.path.exists(p): columns = joblib.load(p); break
    return model, scaler, columns

# ─── Locate files ────────────────────────────────────────────────────────────
csv_path = find_file('youtube_ad_revenue_dataset.csv')
model, scaler, feature_columns = load_model()

# ─── Status Panel ────────────────────────────────────────────────────────────
with st.expander("🔧 File Status — click to expand", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        if csv_path:
            st.success(f"✅ Dataset found: `{csv_path}`")
        else:
            st.error("❌ Dataset NOT found")
            st.info("👉 Place `youtube_ad_revenue_dataset.csv` in the same folder as `app.py`")
    with col2:
        if model:
            st.success(f"✅ Model loaded: `{type(model).__name__}`")
        else:
            st.error("❌ Model NOT found in `model/` folder")
            st.info("👉 Run all 21 cells in the notebook first, then restart Streamlit")

# ─── Load data ───────────────────────────────────────────────────────────────
df = None
if csv_path:
    try:
        df = load_data(csv_path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 EDA Dashboard", "📈 Model Insights"])

# ════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════
with tab1:
    st.markdown("### 🎯 Predict Ad Revenue")

    if model is None or scaler is None or feature_columns is None:
        st.warning("⚠️ Model files not found. Run all 21 cells in the notebook, then restart the app.")
        st.code("streamlit run app.py", language="bash")
    else:
        with st.sidebar:
            st.markdown("## ⚙️ Input Parameters")
            views              = st.slider("👁️ Views",                 5000,    20000,   10000, step=100)
            likes              = st.slider("👍 Likes",                  100,     2500,    1000,  step=50)
            comments           = st.slider("💬 Comments",               10,      600,     200,   step=10)
            watch_time_minutes = st.slider("⏱️ Watch Time (min)",       5000,    80000,   30000, step=1000)
            video_length_min   = st.slider("🎬 Video Length (min)",     1.0,     60.0,    10.0,  step=0.5)
            subscribers        = st.slider("🔔 Subscribers",            1000,    1000000, 500000,step=5000)
            category = st.selectbox("📂 Category", ['Entertainment','Gaming','Education','Music','Tech','Lifestyle'])
            device   = st.selectbox("📱 Device",   ['Mobile','Desktop','Tablet','TV'])
            country  = st.selectbox("🌍 Country",  ['US','IN','CA','UK','DE','AU'])

        engagement_rate = (likes + comments) / views
        avg_watch_time  = watch_time_minutes / views
        like_ratio      = likes / views

        input_data = {col: 0 for col in feature_columns}
        input_data.update({
            'views': views, 'likes': likes, 'comments': comments,
            'watch_time_minutes': watch_time_minutes,
            'video_length_minutes': video_length_min,
            'subscribers': subscribers,
            'engagement_rate': engagement_rate,
            'avg_watch_time': avg_watch_time,
            'like_ratio': like_ratio,
            'month': 6,
        })
        for key in [f'category_{category}', f'device_{device}', f'country_{country}']:
            if key in input_data:
                input_data[key] = 1

        input_df     = pd.DataFrame([input_data])[feature_columns]
        input_scaled = scaler.transform(input_df)
        prediction   = model.predict(input_scaled)[0]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="prediction-box">💰 ${prediction:,.2f} USD</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">📊 Engagement Rate: <b>{engagement_rate:.4f}</b></div>
            <div class="metric-card">⏱️ Avg Watch Time/View: <b>{avg_watch_time:.2f} min</b></div>
            <div class="metric-card">👍 Like Ratio: <b>{like_ratio:.4f}</b></div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("#### 📋 Your Inputs")
            summary = pd.DataFrame({
                'Parameter': ['Views','Likes','Comments','Watch Time','Video Length',
                              'Subscribers','Category','Device','Country'],
                'Value':     [f'{views:,}', f'{likes:,}', f'{comments:,}',
                              f'{watch_time_minutes:,} min', f'{video_length_min} min',
                              f'{subscribers:,}', category, device, country]
            })
            st.dataframe(summary, hide_index=True, use_container_width=True)

# ════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════
with tab2:
    st.markdown("### 📊 Dataset Overview")

    if df is None:
        st.warning("⚠️ Dataset not found. Place `youtube_ad_revenue_dataset.csv` in the same folder as `app.py`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Avg Revenue",   f"${df['ad_revenue_usd'].mean():.2f}")
        c3.metric("Max Revenue",   f"${df['ad_revenue_usd'].max():.2f}")
        c4.metric("Min Revenue",   f"${df['ad_revenue_usd'].min():.2f}")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Revenue Distribution")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(df['ad_revenue_usd'], bins=50, color='#FF0000', alpha=0.8, edgecolor='black')
            ax.set_xlabel('Ad Revenue (USD)'); ax.set_ylabel('Count')
            st.pyplot(fig); plt.close()

        with col2:
            st.markdown("#### Revenue by Category")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            df.groupby('category')['ad_revenue_usd'].mean().sort_values(ascending=False)\
              .plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
            ax.set_ylabel('Avg Revenue (USD)'); ax.tick_params(axis='x', rotation=30)
            st.pyplot(fig); plt.close()

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Revenue by Country")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            df.groupby('country')['ad_revenue_usd'].mean().sort_values(ascending=False)\
              .plot(kind='bar', ax=ax, color='seagreen', edgecolor='black')
            ax.set_ylabel('Avg Revenue (USD)'); ax.tick_params(axis='x', rotation=0)
            st.pyplot(fig); plt.close()

        with col4:
            st.markdown("#### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(6, 4))
            num_cols = ['views','likes','comments','watch_time_minutes',
                        'subscribers','engagement_rate','ad_revenue_usd']
            sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f',
                        cmap='coolwarm', ax=ax, linewidths=0.5, annot_kws={'size':7})
            st.pyplot(fig); plt.close()

# ════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ════════════════════════════════
with tab3:
    st.markdown("### 📈 Model Insights")

    if model is None:
        st.warning("⚠️ Model not found. Run the notebook first.")
    else:
        st.success(f"✅ Active Model: **{type(model).__name__}**")

        if hasattr(model, 'feature_importances_') and feature_columns:
            importances = pd.Series(model.feature_importances_, index=feature_columns)\
                           .sort_values(ascending=False).head(15)
            st.markdown("#### 🏆 Top 15 Feature Importances")
            fig, ax = plt.subplots(figsize=(10, 4))
            importances.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
            ax.set_ylabel('Importance Score'); ax.tick_params(axis='x', rotation=45)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("#### 💡 Key Insights")
        st.markdown("""
        | # | Insight | Finding |
        |---|---|---|
        | 1 | 🥇 Top Driver | `watch_time_minutes` — correlation ~0.99 with revenue |
        | 2 | 🥈 Engagement | `engagement_rate` moderately boosts revenue |
        | 3 | ⚠️ Views ≠ Revenue | Raw views have low correlation — retention matters more |
        | 4 | 🌍 Country CPM | US & CA viewers yield slightly higher ad revenue |
        | 5 | 📱 Device | Mobile viewers generate marginally higher revenue |
        | 6 | 🎯 Strategy | Longer videos + high retention = maximum revenue |
        """)

st.markdown("---")
st.markdown("<center><small>Content Monetization Modeler · Streamlit + Scikit-learn</small></center>",
            unsafe_allow_html=True)
