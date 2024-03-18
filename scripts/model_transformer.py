import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil
from joblib import Parallel, delayed

def create_model_with_transformer_and_train(X_train, y_train, X_test, y_test):
    # Identifying categorical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

    # Creating a ColumnTransformer to apply OneHotEncoding to categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ], remainder='passthrough')  # passthrough numerical columns as is

    # Applying One-Hot Encoding and Data normalization
    X_train = preprocessor.fit_transform(X_train).astype('float16')
    X_test = preprocessor.transform(X_test).astype('float16')

    # Scaling all features (one-hot encoded and numerical)
    scaler = StandardScaler(with_mean=False)  # with_mean=False to support sparse matrix
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape data for Transformer input
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    input_shape = (X_train.shape[1], X_train.shape[2])  # Shape: (1, features)
    
    inputs = layers.Input(shape=input_shape)
    transformer_layer = layers.MultiHeadAttention(num_heads=2, key_dim=2)(inputs, inputs)
    transformer_layer = layers.Dropout(0.1)(transformer_layer)
    transformer_layer = layers.LayerNormalization(epsilon=1e-6)(transformer_layer)
    transformer_output = layers.Flatten()(transformer_layer)
    x = layers.Dense(32, activation='relu')(transformer_output)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2, callbacks=[early_stopping])

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    print('Model summary to verify architecture')
    print(model.summary())

    test_accuracy = accuracy_score(y_test, y_pred)
    print(f'\nTest Accuracy: {test_accuracy}')

    return model, history, test_accuracy

# Note: Make sure to install tabulate if it's not already installed in your Python environment.
# You can do so by running: pip install tabulate
