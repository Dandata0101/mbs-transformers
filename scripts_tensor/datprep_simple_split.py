from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def prepare_and_split_data_simple(df, target_column_name, identifier_columns, test_size=0.2, random_state=42):
    """
    Prepares and splits the data into training and testing sets without grouping.
    """
    print('Separate features, target, and identifiers')
    # Separating out features, target, and identifiers
    X = df.drop(columns=[target_column_name] + identifier_columns)
    feature_names = X.columns.tolist()
    y = df[target_column_name]
    identifiers = df[identifier_columns]

    print('Performing simple train-test split')
    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Extracting identifiers for train and test sets
    identifiers_train, identifiers_test = identifiers.iloc[X_train.index], identifiers.iloc[X_test.index]

    print('Normalizing data (scaling)')
    scaler = StandardScaler()
    # Combining numeric and non-numeric columns after scaling
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    print('Apply scaling only to numeric columns')
    numeric_columns = X_train.select_dtypes(include=['number']).columns
    X_train_scaled[numeric_columns] = scaler.fit_transform(X_train_scaled[numeric_columns])
    X_test_scaled[numeric_columns] = scaler.transform(X_test_scaled[numeric_columns])

    print('Transformation complete')

    # Return the scaled and unscaled data, along with the train and test indices
    return (X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test,
            identifiers_train, identifiers_test, feature_names)
