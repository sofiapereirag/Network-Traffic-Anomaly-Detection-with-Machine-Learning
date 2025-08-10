from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout # type: ignore
from tensorflow.keras import regularizers # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore


def train_model(X_train, y_train, model_type):
    """
    Train a machine learning model based on the specified type.

    Parameters:
    - X_train: Training features.
    - y_train: Training target variable.
    - model_type: Type of model to train ('random_forest', 'decision_tree', 'xgboost').

    Returns:
    - Trained model.
    """
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    elif model_type == 'xgboost':
        # Calculate scale_pos_weight for XGBoost
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = xgb.XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight  # Handle class imbalance
        )
    else:
        raise ValueError("Unsupported model type. Choose from 'random_forest', 'decision_tree', or 'xgboost'.")

    model.fit(X_train, y_train)
    return model



def train_isolation_forest(X,n_estimators, max_samples, contamination, seed):
    """
    Train an Isolation Forest model.
    Parameters:
    - X: DataFrame, features for training.
    - y: Series, target variable (not used in Isolation Forest).
    Returns:
    - model: trained Isolation Forest model.
    """
    model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, random_state=seed)
    model.fit(X)
    print("Isolation Forest model trained.")
    
    return model

def autoencoder(X_train, epochs, batch_size, encoding_dim):
    """
    Train an autoencoder model.
    Parameters:
    - X_train: DataFrame, features for training.
    - epochs: int, number of training epochs.
    - batch_size: int, size of training batches.
    Returns:
    - model: trained autoencoder model.
    """
    input_dim = X_train.shape[1]

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

    autoencoder.fit(X_train, X_train, 
                    epochs=epochs, 
                    batch_size=batch_size,
                    shuffle=True,
                    validation_split=0.2,
                    callbacks=[early_stop, reduce_lr],
                    verbose=1)


    print("Autoencoder model trained.")
    return autoencoder