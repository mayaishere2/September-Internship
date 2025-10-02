import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Suppress verbose TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)


def load_and_preprocess_data(filepath: str) -> tuple:
    """
    Loads the dataset, encodes labels, and scales features.

    Args:
        filepath (str): Path to the dataset CSV file.

    Returns:
        tuple: A tuple containing:
            - df (pd.DataFrame): The preprocessed DataFrame.
            - scaler (MinMaxScaler): The fitted feature scaler.
            - encoder (LabelEncoder): The fitted label encoder.
            - feature_cols (list): The list of feature columns used.
    """
    print(f"[INFO] Loading data from '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"[ERROR] The file '{filepath}' was not found. Please make sure it's in the correct directory.")
        exit()

    # Encode the target labels (fault_type) into integers
    print("[INFO] Encoding fault type labels...")
    encoder = LabelEncoder()
    df['fault_type_encoded'] = encoder.fit_transform(df['fault_type'])
    
    print("Fault Type Mapping:")
    for i, class_name in enumerate(encoder.classes_):
        print(f"  {i}: {class_name}")

    # Select features for the model.
    features_to_scale = df.select_dtypes(include=np.number).columns.tolist()
    features_to_scale = [f for f in features_to_scale if f != 'fault_type_encoded']

    # Scale the features to a range of [0, 1] for the neural network
    print("[INFO] Scaling sensor data features...")
    scaler = MinMaxScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    return df, scaler, encoder, features_to_scale

def create_sliding_windows(df: pd.DataFrame, feature_cols: list, label_col: str, lookback_window: int, prediction_horizon: int) -> tuple:
    """
    Creates sequences of past data (windows) and corresponding future labels.
    """
    print(f"[INFO] Creating sliding windows... (Lookback: {lookback_window} mins, Horizon: {prediction_horizon} mins)")
    X, y = [], []
    # The total span of data needed for one sample is the lookback window plus the time to the future prediction point.
    total_span = lookback_window + prediction_horizon
    
    # Iterate through the dataframe to create sequences
    for i in range(len(df) - total_span):
        # The input window is a sequence of sensor readings from i to i + lookback_window
        feature_sequence = df[feature_cols].iloc[i : i + lookback_window].values
        X.append(feature_sequence)
        
        # The target is the single fault state at the prediction horizon
        target_label = df[label_col].iloc[i + total_span]
        y.append(target_label)
        
    return np.array(X), np.array(y)

def build_lstm_model(input_shape: tuple, num_classes: int) -> tf.keras.Model:
    """
    Builds, compiles, and returns the LSTM model.
    """
    print("[INFO] Building LSTM model...")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax') # Softmax for multi-class classification
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', # Use for integer labels
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def plot_confusion_matrix(y_true, y_pred_classes, class_names):
    """Plots a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Predicted vs. Actual Faults')
    plt.ylabel('Actual Fault')
    plt.xlabel('Predicted Fault')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\n[INFO] Confusion matrix saved as 'confusion_matrix.png'")
    plt.show()

def main(args):
    """Main execution function."""
    
    # 1. Load and prepare the data
    df, scaler, encoder, feature_cols = load_and_preprocess_data(args.dataset)
    
    # 2. Create the time-series sequences for the model
    X, y = create_sliding_windows(
        df,
        feature_cols=feature_cols,
        label_col='fault_type_encoded',
        lookback_window=args.lookback,
        prediction_horizon=args.horizon
    )
    
    if len(X) == 0:
        print("[ERROR] No data windows could be created. The dataset might be too short for the specified lookback and horizon.")
        return

    print(f"\n[INFO] Generated {len(X)} sequences.")
    print(f"  - Input shape: {X.shape}")
    print(f"  - Target shape: {y.shape}")

    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[INFO] Split data into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # 4. Build and train the LSTM model
    num_classes = len(encoder.classes_)
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes)
    
    # Use EarlyStopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("\n[INFO] Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=32,
        validation_split=0.2, # Use part of the training data for validation
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 5. Evaluate the model on the unseen test set
    print("\n[INFO] Evaluating model on the test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  - Test Loss: {loss:.4f}")
    print(f"  - Test Accuracy: {accuracy:.4f}")
    
    # 6. Generate detailed classification report and confusion matrix
    print("\n[INFO] Generating detailed classification report...")
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    report = classification_report(y_test, y_pred_classes, target_names=encoder.classes_)
    print(report)
    
    plot_confusion_matrix(y_test, y_pred_classes, class_names=encoder.classes_)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an LSTM model for predictive maintenance of a gas turbine.")
    parser.add_argument('--dataset', type=str, default='my_turbine_dataset.csv', help="Path to the turbine dataset CSV file.")
    parser.add_argument('--lookback', type=int, default=60, help="Number of minutes of past data to use as input.")
    parser.add_argument('--horizon', type=int, default=120, help="Number of minutes into the future to predict.")
    parser.add_argument('--epochs', type=int, default=25, help="Number of training epochs.")
    
    # MODIFIED: Removed the problematic line that was disabling eager execution.
    main(parser.parse_args())
