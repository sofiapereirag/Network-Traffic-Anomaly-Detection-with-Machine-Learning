from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os


def calculate_pred(model, X_test, y_test, name=None):
    """
    Evaluate the trained model on the test set.
    Parameters:
    - model: Trained machine learning model.
    - X_test: Test features.
    - y_test: Test target variable.
    Returns:
    - y_pred: Predicted labels.
    """
    y_pred = model.predict(X_test)
    if name == 'ae':
        # For autoencoder, we use precision-recall curve
        mse = np.mean((X_test - y_pred)**2, axis=1)  # MSE per sample
        """ # Plot the MSE
        sns.kdeplot(mse[y_test == 0], label="Normal", fill=True)
        sns.kdeplot(mse[y_test == 1], label="Anomaly", fill=True)
        plt.xlabel("Reconstruction Error")
        plt.ylabel("Density")
        plt.legend()
        plt.title("Distribuição do Erro de Reconstrução")
        plt.show() """ 
        precision, recall, thresholds = precision_recall_curve(y_test, mse)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_t = thresholds[np.argmax(f1_scores)]
        print("Best threshold based on F1:", best_t)
        y_pred = (mse > best_t).astype(int) # 1 for attack, 0 for normal
    elif name == 'iso':
        # Predictions: -1 = anomaly, 1 = normal
        # Converting predictions to binary: 1 for attack, 0 for normal
        y_pred = [1 if pred == -1 else 0 for pred in y_pred]  
    
    return y_pred

def evaluate_model(y_test, y_pred):
    """
    Evaluate the model on the test set.
    Parameters:
    - model: Trained machine learning model.
    - x_test: Test features.
    - y_test: Test target variable.
    Returns:
    - metrics: Dictionary containing evaluation metrics.
    """
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, report, cm

def calculate_scores(model, x_test):
    """
    Predict probabilities using the trained model.
    Parameters:
    - model: Trained machine learning model.
    - X_test: Test features.
    Returns:
    - y_proba: Predicted probabilities.
    """
    if hasattr(model, 'predict_proba'):
        y_scores = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        # For Isolation Forest - use decision function
        y_scores = model.decision_function(x_test)
        y_scores = -y_scores # Flip so higher scores indicate anomalies
    else:
        # For Autoencoder - use reconstruction error
        x_reconstructed = model.predict(x_test)
        mse = np.mean((x_test - x_reconstructed)**2, axis=1)
        y_scores = mse  # Higher error = more anomalous

    return y_scores

def evaluate_and_save(model=None, x_test=None, y_test=None, y_pred=None, model_name=None, model_type=None, dataset_name=None, output_path=None):
    """
    Evaluate the model and save the metrics
    Parameters:
    - model: Trained machine learning model.
    - x_test: Test features.
    - y_test: True labels.
    - y_pred: Predicted labels.
    - model_name: Name of the model.
    - dataset_name: Name of the dataset.
    - output_path: Path to save the evaluation report.
    """
    accuracy, report, cm = evaluate_model(y_test, y_pred)

    if(model_name == 'Combined Model (AE + ISO)'):
        # For combined model, we need to handle it differently
        roc_auc = None
        roc_data = None
    else:
        y_scores = calculate_scores(model, x_test)
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = roc_auc_score(y_test, y_scores)
        roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    results = {
        "Model": model_name,
        "ModelType": model_type,
        "Dataset": dataset_name,
        "Report": report,
        "Accuracy": accuracy,
        "Precision": report['1']["precision"],
        "Recall": report['1']["recall"],
        "F1-Score": report['1']["f1-score"],
        "ROC-AUC": roc_auc,
        "ConfusionMatrix": cm.tolist(),
        "ROCData": roc_data
    }

    # Check if file exists and load existing results, or create new list
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                all_results = json.load(f)
            # Ensure all_results is a list
            if not isinstance(all_results, list):
                all_results = [all_results]
        except (json.JSONDecodeError, FileNotFoundError):
            all_results = []
    else:
        all_results = []

    all_results.append(results)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Evaluation results of {model_name} on {dataset_name} saved to {output_path}")

    return results


def print_evaluation_metrics(y_test, y_pred, model_name):
    """
    Print evaluation metrics for the model.
    Parameters:
    - y_test: True labels.
    - y_pred: Predicted labels.
    - model_name: Name of the model.
    """
    accuracy, report, cm = evaluate_model(y_test, y_pred)
    print("Evaluation Metrics for " + model_name)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix of " + model_name)
    plt.show()
    

def combine_predictions(iso_pred, ae_pred):
    """
    Combine predictions from Isolation Forest and Autoencoder.
    Parameters:
    - iso_pred: Predictions from Isolation Forest.
    - ae_pred: Predictions from Autoencoder.
    Returns:
    - combined_pred: Combined predictions.
    """
    return np.logical_or(iso_pred, ae_pred).astype(int)

def predict_combined(ae, iso, threshold, X_ae, X_iso):
    """
    Predict anomalies using both Autoencoder and Isolation Forest.
    Parameters:
    - ae: Trained Autoencoder model.
    - iso: Trained Isolation Forest model.
    - threshold: Threshold for anomaly detection.
    - X: Input features.
    Returns:
    - combined_pred: Combined predictions from both models.
    """
    # Autoencoder prediction
    ae_pred = ae.predict(X_ae)
    mse = np.mean((X_ae - ae_pred)**2, axis=1)
    ae_pred = (mse > threshold).astype(int)

    # Isolation Forest prediction
    iso_pred = iso.predict(X_iso)
    iso_pred = [1 if pred == -1 else 0 for pred in iso_pred]

    # Combine predictions
    combined_pred = np.logical_or(ae_pred, iso_pred).astype(int)
    
    return combined_pred


def plot_roc_curve(model, x_test, y_test, model_name):
    """
    Plot the ROC curve for the model.
    Parameters:
    - metrics: Evaluation metrics for the model.
    """
    y_scores = calculate_scores(model, x_test)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)
    plt.figure()    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - ' + model_name)
    plt.legend()
    plt.grid()
    plt.show()


def plot_recall_comparison(data):
    """
    Plot recall comparison for different models.
    Parameters:
    - data: Data containing model results.
    """
    df = pd.DataFrame(data)
    plt.figure(figsize=(8,5))
    sns.barplot(x='Model', y='Recall', hue='Dataset', data=df)
    plt.title('Recall Comparison by Model and Dataset')
    plt.ylim(0,1)
    plt.ylabel('Recall')
    plt.xlabel('Model')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.show()