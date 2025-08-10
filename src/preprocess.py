# Reusable and clean functions for preprocessing the dataset
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def clean_data(df):
    """
    Clean the dataset by removing unnecessary rows and columns.
    Parameters:
    - df: DataFrame, the dataset to clean.
    Returns:
    - df: DataFrame, cleaned dataset.
    """
    # Remove rows with missing values
    df = df.dropna()
    print("Removed rows with missing values. Remaining rows:", len(df))

    # Remove unnecessary columns
    columns_to_drop = [
        'Flow ID',                 # Unique ID per flow – not useful for prediction
        'Source IP',               # Sensitive data, not generalizable
        'Source Port',             # May have some value but usually very variable    
        'Destination IP',          # Same as Source IP
        'Destination Port',        # Might be useful (e.g., 22 = SSH), but very specific
        'Protocol',                # May choose to keep it for experimentation
        'Timestamp',               # Doesn’t directly contribute to prediction   
        'Label',                   # Original attack name (already converted into 'Attack')
        'Fwd Header Length.1',     # Duplicate of 'Fwd Header Length'
        'Fwd Avg Bytes/Bulk',       # Often zero or null
        'Fwd Avg Packets/Bulk', 
        'Fwd Avg Bulk Rate',   
        'Bwd Avg Bytes/Bulk', 
        'Bwd Avg Packets/Bulk', 
        'Bwd Avg Bulk Rate'
    ]

    df = df.drop(columns = [
        col for col in columns_to_drop if col in df.columns
    ])

    return df

def clean_data_2(X, threshold):
    """
    Clean the dataset by removing unnecessary rows and columns.
    Parameters:
    - X: DataFrame, the dataset to clean.
    Returns:
    - X: DataFrame, cleaned dataset.
    """
    # Make sure all features are numeric
    X = X.select_dtypes(include=[np.number])

    # Apply VarianceThreshold to remove low-variance features
    selector = VarianceThreshold(threshold=threshold)
    selected_mask = selector.fit(X).get_support()
    X = X.loc[:, selected_mask]

    print(f"VarianceThreshold removed {(~selected_mask).sum()} low-variance features")
    print(f"Remaining features: {selected_mask.sum()}")

    return X

def handle_infinite_values(X):
    """
    Handle infinite values in the dataset by replacing them with NaN and then filling those values with median.
    Parameters:
    - X: DataFrame, features to check for infinite values.
    Returns:
    - X_clean: DataFrame, features with infinite values handled.
    """
    # Check for infinite values
    print("Checking for infinite values in the dataset:")
    print(np.isinf(X).sum().sum())

    # Replce infinite values with NaN
    X_clean = X.replace([np.inf, -np.inf], np.nan)

    # Fill NaN values with the median of each column
    X_clean = X_clean.fillna(X_clean.median())
    
    return X_clean

def apply_and_save_scaler(X, scaler_path, scaler=None):
    """
    Scale the features using StandardScaler and saves the scaler.
    Parameters:
    - X: DataFrame, features to scale.
    - scaler_path: str, path to save the fitted scaler.
    Returns:
    - X_scaled: DataFrame, scaled features.
    """
    if scaler == 'minmax':
        scaler = MinMaxScaler()
        print("Using MinMaxScaler for scaling features.")
    else:
        scaler = StandardScaler()
        print("Using StandardScaler for scaling features.")
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, scaler_path)  # Save the scaler for later use
    print(f"Scaler saved as {scaler_path}.")
    
    return X_scaled

def load_scaler_and_transform(X, scaler_path):
    """
    Load the saved scaler and scale the features.
    Parameters:
    - X: DataFrame, features to scale.
    - scaler_path: str, path to the saved scaler.
    Returns:
    - X_scaled: DataFrame, scaled features.
    """
    scaler = joblib.load(scaler_path)

    return scaler.transform(X)

def separate_normal_and_attack(df):
    """
    Separate normal traffic from the dataset.
    Parameters:
    - df: DataFrame, the cleaned dataset.
    Returns:
    - normal_df: DataFrame, containing only normal traffic.
    """
    normal_df = df[df["Attack"] == 0].reset_index(drop=True)
    print(f"Normal traffic rows: {len(normal_df)}")

    return normal_df

def separate_features_and_target(df):
    """
    Separate features and target variable from the dataset.
    Parameters:
    - df: DataFrame, the cleaned dataset.
    Returns:
    - X: DataFrame, features.
    - y: Series, target variable.
    """
    X = df.drop(columns=["Attack"])
    y = df["Attack"]

    return X, y

def split_data(X, y):
    """
    Split the dataset into training and testing sets.
    Parameters:
    - X: DataFrame, normalized features.
    - y: Series, target variable.
    Returns:
    - x_train: DataFrame, training features.
    - x_test: DataFrame, testing features.
    - y_train: Series, training target variable.
    - y_test: Series, testing target variable.
    """
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Data split into training and testing sets.")
    
    return x_train, x_test, y_train, y_test

def apply_and_save_pca(X, n_components, pca_path):
    """
    Apply PCA to reduce dimensionality of the dataset.
    Parameters:
    - X: DataFrame, features to apply PCA on.
    - n_components: int, number of components to keep.
    - pca_path: str, path to save the fitted PCA model.
    Returns:
    - X_pca: DataFrame, transformed features after PCA.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    joblib.dump(pca, pca_path) 
    print(f"PCA applied and saved as {pca_path}. Reduced from {X.shape[1]} to {n_components} components.")

    return X_pca

def load_pca_and_transform(X, pca_path):
    """
    Load the saved PCA model and transform the features.
    Parameters:
    - X: DataFrame, features to transform.
    - pca_path: str, path to the saved PCA model.
    Returns:
    - X_pca: DataFrame, transformed features after PCA.
    """
    pca = joblib.load(pca_path)

    return pca.transform(X)



def resample_data(x_train, y_train):
    """
    Resample the training data with SMOTE to handle class imbalance.
    Parameters:
    - x_train: DataFrame, training features.
    - y_train: Series, training target variable.
    Returns:
    - x_train_resampled: DataFrame, resampled training features.
    - y_train_resampled: Series, resampled training target variable.
    """
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    print("Resampling complete.")
    print("Original training set shape:", x_train.shape)
    print("Resampled training set shape:", x_train_resampled.shape)

    return x_train_resampled, y_train_resampled
