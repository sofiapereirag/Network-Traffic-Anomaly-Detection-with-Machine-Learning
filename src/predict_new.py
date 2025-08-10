from sklearn.metrics import roc_auc_score, roc_curve
import sys
import os
sys.path.append(os.path.abspath("../src"))

from utils import (
    load_data_file,
    load_saved_object
)
from utils_evaluate import (
    calculate_pred,
    calculate_scores,
    evaluate_model,
    predict_combined
)
from preprocess import (
    clean_data,
    handle_infinite_values, 
    clean_data_2,
    separate_features_and_target,
    load_scaler_and_transform,
    load_pca_and_transform
)


def predict_new_data(csv_path, model_path, scaler_path, pca_path, model_name, model_type, dataset_name="Unknown"):

    # ===== Load trained model =====
    model = load_saved_object(model_path)

    # ===== Load and preprocess new data =====
    df = load_data_file(csv_path)
    df = clean_data(df)
    x_test, y_test = separate_features_and_target(df)
    x_test = handle_infinite_values(x_test)
    x_test = clean_data_2(x_test, 0.001)

    # ===== Apply preprocessing based on model type =====
    if model_name == 'Isolation Forest':
        x_test = load_scaler_and_transform(x_test, scaler_path)
        x_test = load_pca_and_transform(x_test, pca_path)
    else:
        x_test = load_scaler_and_transform(x_test, scaler_path)

    # ===== Make predictions =====
    if model_name == 'Isolation Forest':
        y_pred = calculate_pred(model, x_test, y_test, 'iso')
    elif model_name == 'Autoencoder':
        y_pred = calculate_pred(model, x_test, y_test, 'ae')
    else:
        y_pred = calculate_pred(model, x_test, y_test)

    # ===== Evaluate model =====
    accuracy, report, cm = evaluate_model(y_test, y_pred)
    y_scores = calculate_scores(model, x_test)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)
    roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    results = {
        "Model": model_name,
        "Dataset": dataset_name,
        "ModelType": model_type,
        "Report": report,
        "Accuracy": accuracy,
        "Precision": report['1']["precision"],
        "Recall": report['1']["recall"],
        "F1-Score": report['1']["f1-score"],
        "ROC-AUC": roc_auc,
        "ConfusionMatrix": cm.tolist(),
        "ROCData": roc_data
    }    

    # Ensure all_results is a list
    if not isinstance(results, list):
        results = [results]

    return results


def predict_new_data_combined(csv_path, ae_model_path, ae_scaler_path,
                              iso_model_path, iso_scaler_path, iso_pca_path,
                              threshold, dataset_name="Unknown"):
    """
    Predicts anomalies in new data using a combined model (Autoencoder + Isolation Forest).
    Does not calculate ROC, only Accuracy, Report and Confusion Matrix.
    """

    # ===== Load trained models =====
    model_ae = load_saved_object(ae_model_path)
    model_iso = load_saved_object(iso_model_path)

    # ===== Load and preprocess data =====
    df = load_data_file(csv_path)
    df = clean_data(df)
    x_test, y_test = separate_features_and_target(df)
    x_test = handle_infinite_values(x_test)
    x_test = clean_data_2(x_test, 0.001)

    # ===== Preprocessing for AE =====
    x_test_ae = load_scaler_and_transform(x_test, ae_scaler_path)

    # ===== Preprocessing for ISO =====
    x_test_iso = load_scaler_and_transform(x_test, iso_scaler_path)
    x_test_iso = load_pca_and_transform(x_test_iso, iso_pca_path)

    # ====== Make predictions ======
    y_pred = predict_combined(model_ae, model_iso, threshold, x_test_ae, x_test_iso)

    # ====== Evaluate model ======
    accuracy, report, cm = evaluate_model(y_test, y_pred)

    results = {
        "Model": "Combined Model",
        "Dataset": dataset_name,
        "ModelType": "Unsupervised",
        "Report": report,
        "Accuracy": accuracy,
        "Precision": report['1']["precision"],
        "Recall": report['1']["recall"],
        "F1-Score": report['1']["f1-score"],
        "ConfusionMatrix": cm.tolist(),
        "ROC-AUC": None,
        "ROCData": None
    }

    # Ensure all_results is a list
    if not isinstance(results, list):
        results = [results]

    return results