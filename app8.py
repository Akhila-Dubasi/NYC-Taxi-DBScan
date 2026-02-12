import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="NYC Taxi DBSCAN Dashboard", layout="wide")

st.title("🚕 NYC Taxi Pickup Clustering Dashboard")
st.markdown("""
This application uses **DBSCAN clustering** to discover natural pickup location clusters
in NYC taxi trip data.
""")

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configuration Panel")

uploaded_file = st.sidebar.file_uploader("Upload NYC Taxi CSV File", type=["csv"])

row_limit = st.sidebar.slider("Number of Rows to Load", 100, 5000, 500)

eps_value = st.sidebar.slider("Epsilon (eps)", 0.1, 1.0, 0.3, step=0.1)

min_samples = st.sidebar.slider("Minimum Samples", 3, 20, 5)

run_button = st.sidebar.button("Run DBSCAN")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
if uploaded_file:

    df = pd.read_csv(uploaded_file, nrows=row_limit)

    st.subheader("📄 Dataset Preview")
    st.write(df.head())

    required_cols = ['pickup_latitude', 'pickup_longitude']

    if not all(col in df.columns for col in required_cols):
        st.error("Dataset must contain pickup_latitude and pickup_longitude columns.")
        st.stop()

    X = df[required_cols]

    # -----------------------------------------------------
    # SCALING
    # -----------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if run_button:

        # -------------------------------------------------
        # DBSCAN MODEL
        # -------------------------------------------------
        model = DBSCAN(eps=eps_value, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)

        df['Cluster'] = labels

        # -------------------------------------------------
        # CLUSTER METRICS
        # -------------------------------------------------
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_points = list(labels).count(-1)
        noise_ratio = noise_points / len(labels)

        st.subheader("📊 Clustering Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Number of Clusters", n_clusters)
        col2.metric("Noise Points", noise_points)
        col3.metric("Noise Ratio", f"{round(noise_ratio,3)}")

        # -------------------------------------------------
        # SILHOUETTE SCORE
        # -------------------------------------------------
        mask = labels != -1

        if len(set(labels[mask])) > 1:
            score = silhouette_score(X_scaled[mask], labels[mask])
            st.success(f"Silhouette Score: {round(score,3)}")

            if score > 0.5:
                st.info("Clusters are well separated.")
            elif score > 0:
                st.warning("Clusters slightly overlap.")
            else:
                st.error("Poor clustering structure.")
        else:
            st.warning("Silhouette Score: Not Applicable")

        # -------------------------------------------------
        # VISUALIZATION
        # -------------------------------------------------
        st.subheader("🗺 Pickup Location Clusters")

        fig, ax = plt.subplots(figsize=(8,6))

        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                ax.scatter(
                    X.iloc[labels == label]['pickup_longitude'],
                    X.iloc[labels == label]['pickup_latitude'],
                    c='black',
                    label='Noise',
                    alpha=0.6
                )
            else:
                ax.scatter(
                    X.iloc[labels == label]['pickup_longitude'],
                    X.iloc[labels == label]['pickup_latitude'],
                    label=f'Cluster {label}',
                    alpha=0.6
                )

        ax.set_xlabel("Pickup Longitude")
        ax.set_ylabel("Pickup Latitude")
        ax.legend()
        st.pyplot(fig)

        # -------------------------------------------------
        # BUSINESS INSIGHT
        # -------------------------------------------------
        st.subheader("🧠 Business Interpretation")

        st.markdown(f"""
        • The model identified **{n_clusters} natural pickup zones** in NYC.  
        • Approximately **{round(noise_ratio*100,2)}%** of rides occur in isolated locations.  
        • These clusters represent high-demand taxi pickup areas.  

        🚀 This can help:
        - Optimize taxi dispatching
        - Identify high-demand zones
        - Improve fleet allocation
        """)

else:
    st.info("Upload a dataset to begin.")