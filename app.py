# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import sys
import os
sys.path.append(os.path.abspath("src"))
from predict_new import predict_new_data, predict_new_data_combined


st.set_page_config(page_title="Network Anomaly Detection", layout="wide")

# Sidebar for navigation
st.sidebar.header("Settings")

selected_tab = st.sidebar.radio("Choose Tab", ["Predict New Data", "View Metrics"])
metric = st.sidebar.selectbox("Metric", ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"])

if selected_tab == "Predict New Data":

    st.header("Predict New Data")
    # Model selection for prediction
    model_options = {
        "XGBoost": {
            "model_path": "models/xgboost_model.pkl",
            "scaler_path": "models/scalers/2.1_scaler.pkl",
            "pca_path": None,
            "model_type": "Supervised"
        },
        "Random Forest": {
            "model_path": "models/random_forest_model.pkl",
            "scaler_path": "models/scalers/2.1_scaler.pkl",
            "pca_path": None,
            "model_type": "Supervised"
        },
        "Decision Tree": {
            "model_path": "models/decision_tree_model.pkl",
            "scaler_path": "models/scalers/2.1_scaler.pkl",
            "pca_path": None,
            "model_type": "Supervised"
        },
        "Isolation Forest": {
            "model_path": "models/isolation_forest_model2.pkl",
            "scaler_path": "models/scalers/2.2_iso_scaler.pkl",
            "pca_path": "models/pca/pca.pkl",
            "model_type": "Unsupervised"
        },
        "Autoencoder": {
            "model_path": "models/autoencoder_model2.pkl",
            "scaler_path": "models/scalers/2.2_ae_scaler.pkl",
            "pca_path": None,
            "model_type": "Unsupervised"
        },
        "Combined Model (AE + ISO)": {
            "ae_model_path": "models/autoencoder_model2.pkl",
            "iso_model_path": "models/isolation_forest_model2.pkl",
            "ae_scaler_path": "models/scalers/2.2_ae_scaler.pkl",
            "iso_scaler_path": "models/scalers/2.2_iso_scaler.pkl",
            "iso_pca_path": "models/pca/pca.pkl",
            "model_type": "Unsupervised"
        }
    }

    chosen_model = st.selectbox("Choose model", list(model_options.keys()))

    uploaded_file = st.file_uploader(
        f"Upload CSV with new data for {chosen_model}", type="csv"
    )
    

    if uploaded_file is None:
        st.info("No file uploaded yet. Please choose a CSV file to proceed.")
    else:
        cfg = model_options[chosen_model]
        model_type = cfg["model_type"]
        dataset_type = "Unknown" 
        # Predict results
        if chosen_model == "Combined Model (AE + ISO)":
            all_results = predict_new_data_combined(
                uploaded_file,
                cfg["ae_model_path"],
                cfg["ae_scaler_path"],
                cfg["iso_model_path"],
                cfg["iso_scaler_path"],
                cfg["iso_pca_path"],
                threshold=0.00019,  # Adjust as needed
                dataset_name="Unknown"
            )
        else:
            all_results = predict_new_data(
                uploaded_file,
                cfg["model_path"],
                cfg["scaler_path"],
                cfg["pca_path"],
                chosen_model,
                cfg["model_type"],
                "Unknown"
            )

        metrics_list = []
        confusion_matrices = {}
        roc_curves = {}

        for row in all_results:
            metrics_list.append({
                "Model": row["Model"],
                "ModelType": row["ModelType"],
                "Dataset": row["Dataset"],
                "Accuracy": row["Accuracy"],
                "Precision": row["Precision"],
                "Recall": row["Recall"],
                "F1-Score": row["F1-Score"],
                "ROC-AUC": row["ROC-AUC"] if row["ROC-AUC"] is not None else 0.0
            })

            # Keep Confusion Matrix
            if "ConfusionMatrix" in row:
                confusion_matrices[(row["Model"], row["Dataset"])] = row["ConfusionMatrix"]

            # Keep ROC Curve Data
            if "ROCData" in row and row["ROCData"] is not None:
                roc_curves[(row["Model"], row["Dataset"])] = row["ROCData"]

        # Create DataFrame with metrics
        df = pd.DataFrame(metrics_list)

        # Filter DataFrame based on selections
        df = df[(df["ModelType"] == model_type) & (df["Dataset"] == dataset_type)]

        # ==============================
        # METRICS
        # ==============================
        st.subheader(f"{model_type} Models - {dataset_type} Dataset")
        st.dataframe(df, use_container_width=True)

        fig = px.bar(
            df, x="Model", y=metric, color="Model",
            title=f"{metric} Comparison - {model_type} Models ({dataset_type})", text=metric
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)

        # ==============================
        # CONFUSION MATRIX
        # ==============================
        st.subheader("Confusion Matrix")
        selected_model = st.selectbox("Choose Model", df["Model"].unique(), key="tab1_model_select")

        if (selected_model, dataset_type) in confusion_matrices:
            matrix = confusion_matrices[(selected_model, dataset_type)]
            fig_cm = go.Figure(data=go.Heatmap(
                z=matrix,
                x=["Predicted Normal", "Predicted Attack"],
                y=["Actual Normal", "Actual Attack"],
                colorscale="Blues",
                text=matrix,
                texttemplate="%{text}"
            ))
            fig_cm.update_layout(title=f"Confusion Matrix - {selected_model} ({dataset_type})")
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("No confusion matrix available for this model/dataset.")

        # ==============================
        # ROC CURVE
        # ==============================
        st.subheader("ROC Curve")
        if (selected_model, dataset_type) in roc_curves:
            roc_data = roc_curves[(selected_model, dataset_type)]
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=roc_data["fpr"], y=roc_data["tpr"], mode='lines', name='ROC Curve'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash')))
            fig_roc.update_layout(
                title=f"ROC Curve - {selected_model} ({dataset_type})",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                yaxis=dict(range=[0, 1]),
                xaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("No ROC curve available for this model/dataset.")
else:

    st.header("Model Evaluation Metrics")

    # Filter based on selections
    model_type = st.sidebar.radio("ModelType", ["Supervised", "Unsupervised"])
    dataset_type = st.sidebar.radio("Dataset", ["Known", "Unknown"])

    # Load results
    results_path = "results/metrics.json"
    with open(results_path, "r") as f:
        all_results = json.load(f)
    print(type(all_results)) 

    metrics_list = []
    confusion_matrices = {}
    roc_curves = {}

    for row in all_results:
            metrics_list.append({
                "Model": row["Model"],
                "ModelType": row["ModelType"],
                "Dataset": row["Dataset"],
                "Accuracy": row["Accuracy"],
                "Precision": row["Precision"],
                "Recall": row["Recall"],
                "F1-Score": row["F1-Score"],
                "ROC-AUC": row["ROC-AUC"] if row["ROC-AUC"] is not None else 0.0
            })

            # Keep Confusion Matrix
            if "ConfusionMatrix" in row:
                confusion_matrices[(row["Model"], row["Dataset"])] = row["ConfusionMatrix"]

            # Keep ROC Curve Data
            if "ROCData" in row and row["ROCData"] is not None:
                roc_curves[(row["Model"], row["Dataset"])] = row["ROCData"]

    # Create DataFrame with metrics
    df = pd.DataFrame(metrics_list)

    # Filter DataFrame based on selections
    df = df[(df["ModelType"] == model_type) & (df["Dataset"] == dataset_type)]

    # ==============================
    # METRICS
    # ==============================
    st.subheader(f"{model_type} Models - {dataset_type} Dataset")
    st.dataframe(df, use_container_width=True)

    fig = px.bar(
        df, x="Model", y=metric, color="Model",
        title=f"{metric} Comparison - {model_type} Models ({dataset_type})", text=metric
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # CONFUSION MATRIX
    # ==============================
    st.subheader("Confusion Matrix")
    selected_model = st.selectbox("Choose Model", df["Model"].unique(), key="tab2_model_select")

    if (selected_model, dataset_type) in confusion_matrices:
        matrix = confusion_matrices[(selected_model, dataset_type)]
        fig_cm = go.Figure(data=go.Heatmap(
            z=matrix,
            x=["Predicted Normal", "Predicted Attack"],
            y=["Actual Normal", "Actual Attack"],
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}"
        ))
        fig_cm.update_layout(title=f"Confusion Matrix - {selected_model} ({dataset_type})")
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("No confusion matrix available for this model/dataset.")

    # ==============================
    # ROC CURVE
    # ==============================
    st.subheader("ROC Curve")
    if (selected_model, dataset_type) in roc_curves:
        roc_data = roc_curves[(selected_model, dataset_type)]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=roc_data["fpr"], y=roc_data["tpr"], mode='lines', name='ROC Curve'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash')))
        fig_roc.update_layout(
            title=f"ROC Curve - {selected_model} ({dataset_type})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            yaxis=dict(range=[0, 1]),
            xaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.info("No ROC curve available for this model/dataset.")
