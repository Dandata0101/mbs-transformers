import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import datetime

def create_model_with_transformer_and_train(X_train, y_train, X_test, y_test):
    # Identifying categorical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

    # Creating a ColumnTransformer to apply OneHotEncoding to categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ], remainder='passthrough')  # Passthrough numerical columns as is

    # Applying One-Hot Encoding
    X_train_encoded = preprocessor.fit_transform(X_train).astype('float32')
    X_test_encoded = preprocessor.transform(X_test).astype('float32')

    # Scaling all features (one-hot encoded and numerical)
    scaler = StandardScaler(with_mean=False)  # with_mean=False to support sparse matrix
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    # Reshape data for Transformer input
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])  # Shape: (1, features)
    
    inputs = layers.Input(shape=input_shape)
    transformer_layer = layers.MultiHeadAttention(num_heads=2, key_dim=2)(inputs, inputs)
    transformer_layer = layers.Dropout(0.1)(transformer_layer)
    transformer_layer = layers.LayerNormalization(epsilon=1e-6)(transformer_layer)
    transformer_output = layers.Flatten()(transformer_layer)
    x = layers.Dense(32, activation='relu')(transformer_output)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # TensorBoard callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)
    
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Training the model
    history = model.fit(X_train_reshaped, y_train, epochs=5, batch_size=128, validation_split=0.2, callbacks=[early_stopping, tensorboard_callback])

    y_pred_prob = model.predict(X_test_reshaped)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    test_accuracy = accuracy_score(y_test, y_pred)
    print(f'\nTest Accuracy: {test_accuracy}')

    return model, history, test_accuracy,y_test, y_pred,X_train_reshaped,X_test_reshaped
