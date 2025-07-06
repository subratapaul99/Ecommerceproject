import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap

import os
import gdown

# Google Drive direct download links
model_url = "https://drive.google.com/uc?id=1XxYRj9M41EU9InrTU9k-jRMvqUU2mEF0"
scaler_url = "https://drive.google.com/uc?id=1cAxpJvhow2XF5-hwRn5-JeI3GyqeMLFc"
feature_url = "https://drive.google.com/uc?id=12f4TL4g-iFjiytNknwm98sIb4g4aCJcL"

# Auto-download if files are missing
if not os.path.exists("sales_prediction_model.pkl"):
    gdown.download(model_url, "sales_prediction_model.pkl", quiet=False)

if not os.path.exists("scaler.pkl"):
    gdown.download(scaler_url, "scaler.pkl", quiet=False)

if not os.path.exists("feature_columns.pkl"):
    gdown.download(feature_url, "feature_columns.pkl", quiet=False)


# Load trained model and preprocessing tools
model = joblib.load("sales_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="🧠 Sales AI Dashboard", layout="wide")
st.title("📊 AI Sales Prediction Dashboard")

# Tabs for clean navigation
tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "📤 Upload & Predict", "🔮 Manual Prediction"])

# ===============================
# 📈 Tab 1: EDA Dashboard
# ===============================
with tab1:
    st.subheader("📊 Interactive Sales Dashboard")

    try:
        df = pd.read_csv("Sample - Superstore.csv", encoding="latin-1")

        drop_cols = ['Row ID', 'Order ID', 'Customer ID', 'Customer Name', 'Product ID', 'Product Name']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        df["Order Date"] = pd.to_datetime(df["Order Date"])

        # Sidebar filters
        st.sidebar.header("🔍 Filter Options")
        region_filter = st.sidebar.multiselect("Select Region", options=df["Region"].unique(), default=df["Region"].unique())
        category_filter = st.sidebar.multiselect("Select Category", options=df["Category"].unique(), default=df["Category"].unique())
        date_range = st.sidebar.date_input("Select Date Range", [df["Order Date"].min(), df["Order Date"].max()])

        # Apply filters
        mask = (
            df["Region"].isin(region_filter) &
            df["Category"].isin(category_filter) &
            (df["Order Date"] >= pd.to_datetime(date_range[0])) &
            (df["Order Date"] <= pd.to_datetime(date_range[1]))
        )
        filtered_df = df[mask]

        # KPIs
        total_sales = filtered_df["Sales"].sum()
        avg_profit = filtered_df["Profit"].mean()
        top_city = filtered_df.groupby("City")["Sales"].sum().idxmax()

        k1, k2, k3 = st.columns(3)
        k1.metric("💰 Filtered Sales", f"₹{total_sales:,.0f}")
        k2.metric("📈 Avg. Profit", f"₹{avg_profit:,.0f}")
        k3.metric("🏙️ Top City", top_city)

        # Sales Over Time
        st.markdown("### 🕒 Sales Over Time")
        sales_time = filtered_df.groupby("Order Date")["Sales"].sum().reset_index()
        fig_time = px.line(sales_time, x="Order Date", y="Sales", title="Sales Trend")
        st.plotly_chart(fig_time, use_container_width=True)

        # Heatmap
        st.markdown("### 🔥 Sales Heatmap (Region vs Category)")
        heatmap_data = filtered_df.groupby(["Region", "Category"])["Sales"].sum().reset_index()
        fig_heatmap = px.density_heatmap(heatmap_data, x="Region", y="Category", z="Sales", color_continuous_scale="Viridis")
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Sub-category bar
        st.markdown("### 🧾 Sales by Sub-Category")
        subcat_data = filtered_df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False).reset_index()
        fig_subcat = px.bar(subcat_data, x="Sub-Category", y="Sales", color="Sales", title="Sub-Category Sales")
        st.plotly_chart(fig_subcat, use_container_width=True)

        # Scatter plot
        st.markdown("### 💸 Profit vs Sales Scatter Plot")
        if st.checkbox("📈 Show Scatter Plot"):
            fig_scatter = px.scatter(filtered_df, x="Sales", y="Profit", color="Category",
                                     size="Quantity", hover_data=["City", "Sub-Category"],
                                     title="Profit vs Sales by Category")
            st.plotly_chart(fig_scatter, use_container_width=True)

    except Exception as e:
        st.error(f"Dashboard loading failed: {e}")

# ===============================
# 📤 Tab 2: Upload & Predict
# ===============================
with tab2:
    st.subheader("📤 Upload a CSV File to Predict Sales")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            encoded = pd.get_dummies(data)
            encoded = encoded.reindex(columns=feature_columns, fill_value=0)

            scaled = scaler.transform(encoded)
            predictions = model.predict(scaled)

            data["Predicted Sales"] = predictions
            st.success("✅ Prediction completed successfully!")
            st.dataframe(data)

            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Prediction Results", csv, "predicted_sales.csv", "text/csv")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ===============================
# 🔮 Tab 3: Manual Prediction
# ===============================
with tab3:
    st.subheader("🔮 Manual Sales Prediction")

    region = st.selectbox("Region", ["Central", "East", "South", "West"])
    category = st.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])
    sub_category = st.selectbox("Sub-Category", ["Chairs", "Phones", "Binders", "Paper", "Storage"])
    segment = st.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])
    ship_mode = st.selectbox("Ship Mode", ["Second Class", "Standard Class", "First Class", "Same Day"])
    quantity = st.slider("Quantity", 1, 20, 5)
    discount = st.slider("Discount", 0.0, 0.9, 0.1)

    input_df = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
    input_df["Quantity"] = quantity
    input_df["Discount"] = discount

    for col in [f"Region_{region}", f"Category_{category}", f"Sub-Category_{sub_category}",
                f"Segment_{segment}", f"Ship Mode_{ship_mode}"]:
        if col in input_df.columns:
            input_df[col] = 1

    if st.button("🔮 Predict Now"):
        try:
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            st.success(f"💰 Predicted Sales: ₹{prediction:.2f}")

            # SHAP Explainability
            explainer = shap.Explainer(model, feature_names=input_df.columns)
            shap_values = explainer(input_scaled)

            st.markdown("### 🧠 SHAP Feature Importance")
            shap_df = pd.DataFrame({
                "Feature": input_df.columns,
                "SHAP Value": shap_values[0].values,
                "Input Value": input_df.iloc[0].values
            }).sort_values(by="SHAP Value", key=abs, ascending=False)

            st.dataframe(shap_df)

            fig_shap = px.bar(shap_df.head(10), x="SHAP Value", y="Feature", orientation="h",
                              title="Top Contributing Features", color="SHAP Value")
            st.plotly_chart(fig_shap, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction or SHAP failed: {e}")

# ================================
# Footer
# ================================
st.markdown("---")
st.caption("🚀 Built with ❤️ by Subrata | Powered by Streamlit + scikit-learn")
