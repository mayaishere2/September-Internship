import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow (only if you use TF ops, device management, etc.)
import tensorflow as tf

# Use standalone keras (Keras 3)
from keras.models import Model, Sequential, load_model
from keras.layers import Input, LSTM, Dense, AdditiveAttention, Bidirectional, RepeatVector, Concatenate, TimeDistributed
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib


# Suppress verbose TensorFlow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)


def load_and_preprocess_data(filepath: str) -> tuple:
    """
    Loads the dataset, encodes labels, and scales features.
    """
    print(f"[INFO] Loading data from '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"[ERROR] The file '{filepath}' was not found.")
        exit()

    encoder = LabelEncoder()
    df['fault_type_encoded'] = encoder.fit_transform(df['fault_type'])
    
    print("Fault Type Mapping:")
    for i, class_name in enumerate(encoder.classes_):
        print(f"  {i}: {class_name}")

    features_to_scale = df.select_dtypes(include=np.number).columns.tolist()
    features_to_scale = [f for f in features_to_scale if f != 'fault_type_encoded']

    scaler = MinMaxScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    return df, scaler, encoder, features_to_scale

def create_forecasting_windows(df: pd.DataFrame, feature_cols: list, lookback: int, horizon: int) -> tuple:
    """
    Creates sequences for a seq2seq forecasting model.
    Input X: A sequence of past data.
    Output y: A sequence of future data.
    """
    print(f"[INFO] Creating forecasting windows... (Lookback: {lookback} mins, Horizon: {horizon} mins)")
    X, y = [], []
    total_steps = lookback + horizon
    
    for i in range(len(df) - total_steps + 1):
        past_indices = range(i, i + lookback)
        future_indices = range(i + lookback, i + lookback + horizon)
        
        X.append(df[feature_cols].iloc[past_indices].values)
        y.append(df[feature_cols].iloc[future_indices].values)
        
    return np.array(X), np.array(y)

# MODIFIED: Replaced the previous model with a deeper, bidirectional architecture for higher accuracy.
def build_forecasting_model(lookback: int, horizon: int, num_features: int) -> Model:
    """
    Builds a more powerful Bidirectional LSTM Encoder-Decoder model with Attention.
    """
    print("[INFO] Building a deeper, Bidirectional Attention-based LSTM forecasting model...")
    
    # --- Encoder ---
    encoder_inputs = Input(shape=(lookback, num_features), name='encoder_inputs')
    # Bidirectional LSTM captures patterns from both forward and backward passes.
    encoder_bilstm = Bidirectional(LSTM(128, return_sequences=True, return_state=True, name='encoder_lstm_1'))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_bilstm(encoder_inputs)
    
    # Concatenate the forward and backward states to initialize the decoder
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    # --- Decoder ---
    # The decoder input will be the last value of the encoder input sequence, repeated 'horizon' times.
    decoder_input_seed = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(encoder_inputs)
    decoder_inputs = RepeatVector(horizon)(decoder_input_seed)

    # Decoder LSTM needs to have units compatible with the concatenated encoder states (128*2=256)
    decoder_lstm = LSTM(256, return_sequences=True, name='decoder_lstm')
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # --- Attention Mechanism ---
    attention_layer = AdditiveAttention(name='attention_layer')
    context_vector = attention_layer([decoder_outputs, encoder_outputs])
    decoder_combined_context = Concatenate(axis=-1)([context_vector, decoder_outputs])

    # Final prediction layer for each time step.
    output_layer = Dense(num_features, name='output_dense')
    outputs = TimeDistributed(output_layer)(decoder_combined_context)

    # --- Build and Compile Model ---
    model = Model(inputs=encoder_inputs, outputs=outputs, name='bidirectional_attention_seq2seq')
    # MODIFIED: Using Mean Absolute Error and a fine-tuned Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mae') # MAE is less sensitive to outliers and can produce sharper forecasts
    model.summary()
    return model


def plot_forecast_vs_actual(y_true, y_pred, feature_cols, scaler, sample_idx=0):
    """Plots a comparison of forecasted vs actual values for key features."""
    
    # Inverse transform the entire batch to get original scale
    y_true_rescaled = scaler.inverse_transform(y_true[sample_idx])
    y_pred_rescaled = scaler.inverse_transform(y_pred[sample_idx])

    # Select specific, important features to plot for better insight
    features_to_plot = ['TIT', 'CDP', 'T_bearing', 'V_shaft']
    
    plt.figure(figsize=(15, 10))
    
    for i, feature_name in enumerate(features_to_plot):
        if feature_name not in feature_cols:
            print(f"[WARN] Feature '{feature_name}' not in columns, skipping plot.")
            continue
            
        feature_idx = feature_cols.index(feature_name)
        
        plt.subplot(len(features_to_plot), 1, i + 1)
        plt.plot(y_true_rescaled[:, feature_idx], label='Actual', color='blue', marker='o', markersize=4)
        plt.plot(y_pred_rescaled[:, feature_idx], label='Forecasted', color='red', linestyle='--')
        plt.title(f'Forecast vs. Actual for: {feature_name}')
        plt.ylabel('Value')
        plt.legend()

    plt.xlabel('Time Steps into Future (Minutes)')
    plt.tight_layout()
    plt.savefig('forecast_vs_actual.png')
    print("\n[INFO] Forecast plot saved as 'forecast_vs_actual.png'")
    
def main(args):
    # 1. Load Data
    df, scaler, encoder, feature_cols = load_and_preprocess_data(args.dataset)
    
    # --- NEW: SAVE PREPROCESSING OBJECTS ---
    joblib.dump(scaler, args.save_scaler)
    joblib.dump(encoder, args.save_encoder)
    print(f"[INFO] Scaler saved to '{args.save_scaler}'")
    print(f"[INFO] Encoder saved to '{args.save_encoder}'")
    
    # 2. Create Windows for the Forecasting Model
    X_forecast, y_forecast = create_forecasting_windows(df, feature_cols, args.lookback, args.horizon)
    
    # 3. Create Labels for the Classification Model
    y_classify = []
    total_steps = args.lookback + args.horizon
    for i in range(len(df) - total_steps + 1):
        y_classify.append(df['fault_type_encoded'].iloc[i + total_steps - 1])
    y_classify = np.array(y_classify)
    
    if len(X_forecast) == 0:
        print("[ERROR] No data windows created. Dataset too short for the lookback/horizon.")
        return

    # 4. Split data for both models
    indices = np.arange(X_forecast.shape[0])
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_classify)

    X_forecast_train, X_forecast_test = X_forecast[train_indices], X_forecast[test_indices]
    y_forecast_train, y_forecast_test = y_forecast[train_indices], y_forecast[test_indices]
    y_classify_train, y_classify_test = y_classify[train_indices], y_classify[test_indices]

    # --- PART 1: TRAIN THE FORECASTING MODEL ---
    forecaster = build_forecasting_model(args.lookback, args.horizon, len(feature_cols))
    
    # MODIFIED: Increased patience and added a learning rate scheduler
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    print("\n[INFO] Training the forecasting model...")
    forecaster.fit(
        X_forecast_train, y_forecast_train,
        epochs=args.epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr], # Added ReduceLROnPlateau
        verbose=1
    )
    # --- NEW: SAVE FORECASTING MODEL ---
    forecaster.save(args.save_forecaster)
    print(f"\n[INFO] Forecasting model saved to '{args.save_forecaster}'")

    # --- PART 2: USE FORECASTER TO CREATE DATA FOR CLASSIFIER ---
    print("\n[INFO] Using forecaster to generate features for the classification model...")
    train_forecasts = forecaster.predict(X_forecast_train)
    X_classify_train = train_forecasts[:, -1, :] 
    
    test_forecasts = forecaster.predict(X_forecast_test)
    X_classify_test = test_forecasts[:, -1, :]

    # --- PART 3: TRAIN THE CLASSIFICATION MODEL ---
    print("\n[INFO] Training the Random Forest classifier...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    classifier.fit(X_classify_train, y_classify_train)
    
    # --- NEW: SAVE CLASSIFICATION MODEL ---
    joblib.dump(classifier, args.save_classifier)
    print(f"[INFO] Classification model saved to '{args.save_classifier}'")

    # --- PART 4: EVALUATE BOTH MODELS ---
    print("\n--- FORECASTING MODEL EVALUATION ---")
    mae = forecaster.evaluate(X_forecast_test, y_forecast_test, verbose=0)
    print(f"  - Forecasting Test MAE: {mae:.6f}") # Switched to MAE
    plot_forecast_vs_actual(y_forecast_test, test_forecasts, feature_cols, scaler, sample_idx=np.random.randint(0, len(y_forecast_test)))
    
    print("\n--- CLASSIFICATION MODEL EVALUATION ---")
    y_pred_classify = classifier.predict(X_classify_test)
    report = classification_report(y_classify_test, y_pred_classify, target_names=encoder.classes_)
    print(report)

    cm = confusion_matrix(y_classify_test, y_pred_classify)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Classifier Confusion Matrix (on Forecasted Data)')
    plt.ylabel('Actual Fault')
    plt.xlabel('Predicted Fault')
    plt.tight_layout()
    plt.savefig('classifier_confusion_matrix.png')
    print("\n[INFO] Classifier confusion matrix saved as 'classifier_confusion_matrix.png'")
    
    # --- PART 4.5: SAVE PREDICTIONS TO CSV ---
    print("\n[INFO] Saving test set predictions to a CSV file...")
    
    # Inverse transform the forecasted features to their original scale
    forecasted_features_rescaled = scaler.inverse_transform(X_classify_test)
    
    # Inverse transform the labels from integers back to string names
    actual_labels = encoder.inverse_transform(y_classify_test)
    predicted_labels = encoder.inverse_transform(y_pred_classify)
    
    # Create a DataFrame for the results
    results_df = pd.DataFrame(forecasted_features_rescaled, columns=[f"{col}_forecasted" for col in feature_cols])
    results_df['actual_fault'] = actual_labels
    results_df['predicted_fault'] = predicted_labels
    
    # Reorder columns to have labels first for clarity
    cols_to_move = ['actual_fault', 'predicted_fault']
    results_df = results_df[cols_to_move + [col for col in results_df.columns if col not in cols_to_move]]
    
    # Save to CSV
    results_df.to_csv(args.output, index=False)
    print(f"[INFO] Predictions saved to '{args.output}'")
    print("Preview of the predictions file:")
    print(results_df.head())


    # --- PART 5: DEMONSTRATE PREDICTION AND ALERTING ---
    print("\n--- PREDICTION & ALERTING DEMO ---")
    sample_index = np.random.randint(0, len(X_forecast_test))
    new_data_sequence = X_forecast_test[sample_index:sample_index+1]
    actual_future_fault_code = y_classify_test[sample_index]
    actual_fault_name = encoder.inverse_transform([actual_future_fault_code])[0]

    print(f"Step 1: Forecasting future sensor values...")
    predicted_future_sequence = forecaster.predict(new_data_sequence)
    predicted_future_state = predicted_future_sequence[:, -1, :]

    print(f"Step 2: Classifying the forecasted future state...")
    predicted_fault_code = classifier.predict(predicted_future_state)[0]
    predicted_fault_name = encoder.inverse_transform([predicted_fault_code])[0]

    print("\n--- RESULT ---")
    print(f"  - Actual Fault in 2 hours will be: '{actual_fault_name}'")
    print(f"  - Model Prediction for 2 hours from now is: '{predicted_fault_name}'")

    if predicted_fault_name != 'normal':
        print(f"\n  ************************************************")
        print(f"  * ALERT! Predicted '{predicted_fault_name}'    *")
        print(f"  * fault in the next {args.horizon} minutes.          *")
        print(f"  ************************************************")
    else:
        print("\n  - System state predicted to be normal.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Two-step predictive maintenance: Forecast sensor values then classify the future state.")
    parser.add_argument('--dataset', type=str, default='my_turbine_dataset.csv', help="Path to the turbine dataset.")
    parser.add_argument('--lookback', type=int, default=60, help="Minutes of past data to use as input.")
    parser.add_argument('--horizon', type=int, default=120, help="Minutes into the future to forecast and predict.")
    parser.add_argument('--epochs', type=int, default=75, help="Number of training epochs for the forecaster.")
    # NEW: Added an argument for the output file path
    parser.add_argument('--output', type=str, default='forecast_predictions.csv', help="Path for the output CSV file with predictions.")
    # NEW: Arguments for saving the trained models and preprocessing objects
    parser.add_argument('--save_forecaster', type=str, default='forecasting_model.keras', help="Path to save the trained forecasting model.")
    parser.add_argument('--save_classifier', type=str, default='classification_model.joblib', help="Path to save the trained classification model.")
    parser.add_argument('--save_scaler', type=str, default='scaler.joblib', help="Path to save the fitted scaler.")
    parser.add_argument('--save_encoder', type=str, default='encoder.joblib', help="Path to save the fitted label encoder.")

    main(parser.parse_args())

