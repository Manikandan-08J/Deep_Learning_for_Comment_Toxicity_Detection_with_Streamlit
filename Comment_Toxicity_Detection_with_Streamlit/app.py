
import os, json, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.predict import load_artifacts, predict_single, predict_batch

st.set_page_config(page_title="Toxicity Detector", page_icon="🛡️", layout="wide")

LABEL_COLS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
THRESHOLD  = 0.5

@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_artifacts()

page = st.sidebar.radio("Navigate", [
    "🔍 Single Prediction",
    "📂 Bulk CSV Prediction",
    "📊 EDA Dashboard",
    "📈 Model Metrics"
])
st.sidebar.markdown("---")
st.sidebar.caption("Deep Learning · NLP · Streamlit")

def prob_bar_chart(result):
    probs  = [result[l] * 100 for l in LABEL_COLS]
    colors = ["#e74c3c" if p >= 50 else "#27ae60" for p in probs]
    fig = go.Figure(go.Bar(x=LABEL_COLS, y=probs, marker_color=colors,
                           text=[f"{p:.1f}%" for p in probs], textposition="outside"))
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Threshold 50%")
    fig.update_layout(title="Toxicity probability by category",
                      yaxis=dict(range=[0,115], title="Probability (%)"), height=380)
    return fig

def gauge(score, label):
    color = "#e74c3c" if score >= THRESHOLD else "#27ae60"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(score*100, 1),
        title={"text": label.replace("_"," ").title()},
        gauge={"axis":{"range":[0,100]}, "bar":{"color":color},
               "steps":[{"range":[0,50],"color":"#d5f5e3"},
                        {"range":[50,80],"color":"#fdebd0"},
                        {"range":[80,100],"color":"#fadbd8"}],
               "threshold":{"line":{"color":"black","width":4},"value":50,"thickness":0.75}},
        number={"suffix":"%"}
    ))
    fig.update_layout(height=200, margin=dict(t=30,b=0,l=10,r=10))
    return fig

if page == "🔍 Single Prediction":
    st.title("🔍 Single Comment Prediction")
    comment = st.text_area("Enter a comment", height=150, placeholder="Type here...")
    if st.button("🚀 Predict"):
        if not comment.strip():
            st.warning("Please enter a comment.")
        else:
            model, tokenizer = get_model()
            result = predict_single(comment, model, tokenizer)
            if result["is_toxic"]:
                st.error(f"⚠️ Toxic — dominant: **{result['dominant_label'].replace('_',' ').title()}**")
            else:
                st.success("✅ Clean comment — no toxicity detected.")
            st.plotly_chart(prob_bar_chart(result), use_container_width=True)
            st.subheader("Per-category confidence")
            cols = st.columns(3)
            for i, label in enumerate(LABEL_COLS):
                with cols[i % 3]:
                    st.plotly_chart(gauge(result[label], label), use_container_width=True, key=label)
            with st.expander("Raw scores"):
                st.dataframe(pd.DataFrame({
                    "Category":    [l.replace("_"," ").title() for l in LABEL_COLS],
                    "Probability": [f"{result[l]*100:.2f}%" for l in LABEL_COLS],
                    "Flagged":     ["✅ Yes" if result[l] >= THRESHOLD else "❌ No" for l in LABEL_COLS]
                }), use_container_width=True, hide_index=True)

elif page == "📂 Bulk CSV Prediction":
    st.title("📂 Bulk CSV Prediction")
    st.markdown("Upload a CSV with a **comment_text** column.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_in = pd.read_csv(uploaded)
        if "comment_text" not in df_in.columns:
            st.error("CSV must have a comment_text column.")
        else:
            st.write(f"**{len(df_in):,} comments** loaded.")
            st.dataframe(df_in.head(5), use_container_width=True)
            if st.button("🚀 Run Predictions"):
                model, tokenizer = get_model()
                progress = st.progress(0, text="Predicting...")
                texts, results, chunk = df_in["comment_text"].fillna("").tolist(), [], 1000
                for i in range(0, len(texts), chunk):
                    results.append(predict_batch(texts[i:i+chunk], model, tokenizer))
                    progress.progress(min((i+chunk)/len(texts), 1.0))
                df_out = pd.concat(results, ignore_index=True)
                if "id" in df_in.columns:
                    df_out.insert(0, "id", df_in["id"].values)
                progress.empty()
                st.success(f"Done! **{df_out['is_toxic'].sum():,}** toxic comments found.")
                counts = df_out[LABEL_COLS].apply(lambda c: (c >= THRESHOLD).sum())
                st.plotly_chart(px.bar(x=counts.index, y=counts.values,
                                       title="Toxic count per category",
                                       labels={"x":"Category","y":"Count"}),
                                use_container_width=True)
                st.dataframe(df_out.head(20), use_container_width=True)
                st.download_button("⬇️ Download Results CSV",
                                   data=df_out.to_csv(index=False).encode(),
                                   file_name="predictions.csv", mime="text/csv")

elif page == "📊 EDA Dashboard":
    st.title("📊 EDA Dashboard")
    if not os.path.exists("data/train.csv"):
        st.warning("Place train.csv in the data/ folder.")
    else:
        @st.cache_data
        def load():
            return pd.read_csv("data/train.csv")
        df = load()
        st.markdown(f"**{len(df):,} comments · 6 toxicity labels**")
        counts = df[LABEL_COLS].sum().reset_index()
        counts.columns = ["Category","Count"]
        counts["Pct"] = (counts["Count"]/len(df)*100).round(2)
        fig1 = px.bar(counts, x="Category", y="Count", text="Pct",
                      title="Label distribution", color="Category")
        fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig1, use_container_width=True)
        any_toxic = (df[LABEL_COLS].sum(axis=1) > 0).sum()
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.pie(names=["Clean","Toxic"],
                                   values=[len(df)-any_toxic, any_toxic],
                                   title="Clean vs Toxic",
                                   color_discrete_sequence=["#27ae60","#e74c3c"]),
                            use_container_width=True)
        with col2:
            df["char_len"] = df["comment_text"].fillna("").apply(len)
            st.plotly_chart(px.histogram(df, x="char_len", nbins=80,
                                         title="Comment length",
                                         labels={"char_len":"Characters"},
                                         color_discrete_sequence=["#3498db"],
                                         range_x=[0,2000]),
                            use_container_width=True)
        st.subheader("Label co-occurrence heatmap")
        st.plotly_chart(px.imshow(df[LABEL_COLS].corr(), text_auto=".2f",
                                  color_continuous_scale="RdBu_r",
                                  title="Pearson correlation"),
                        use_container_width=True)

elif page == "📈 Model Metrics":
    st.title("📈 Model Metrics")
    if not os.path.exists("models/metrics.json"):
        st.info("Train the model first. Metrics will appear here automatically.")
    else:
        with open("models/metrics.json") as f:
            m = json.load(f)
        auc_df = pd.DataFrame({"Category": list(m["auc"].keys()),
                                "AUC-ROC":  list(m["auc"].values())})
        st.plotly_chart(px.bar(auc_df, x="Category", y="AUC-ROC", color="Category",
                               title="AUC-ROC per label", range_y=[0.9, 1.0]),
                        use_container_width=True)
        hist   = m["history"]
        epochs = list(range(1, len(hist["loss"]) + 1))
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=hist["loss"],     name="Train",      line=dict(color="#e74c3c")))
            fig.add_trace(go.Scatter(x=epochs, y=hist["val_loss"], name="Validation", line=dict(color="#3498db", dash="dash")))
            fig.update_layout(title="Loss", xaxis_title="Epoch", yaxis_title="Loss", height=300)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=epochs, y=hist["accuracy"],     name="Train",      line=dict(color="#27ae60")))
            fig2.add_trace(go.Scatter(x=epochs, y=hist["val_accuracy"], name="Validation", line=dict(color="#f39c12", dash="dash")))
            fig2.update_layout(title="Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy", height=300)
            st.plotly_chart(fig2, use_container_width=True)
