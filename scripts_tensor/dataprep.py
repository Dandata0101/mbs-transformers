from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def prepare_and_split_data_grouped(df, target_column_name, identifier_columns, group_column, test_size=0.2, random_state=42):
    """
    Prepares and splits the data into training and testing sets based on groups.
    It ensures that data within the same group is not split across training and testing sets.
    """
    print('Separate features, target, and identifiers, ensuring group column is included in identifiers')
    if group_column not in identifier_columns:
        identifier_columns.append(group_column)
    
    # Separating out features, target, and identifiers
    X = df.drop(columns=[target_column_name] + identifier_columns)
    feature_names = X.columns.tolist()
    y = df[target_column_name]
    identifiers = df[identifier_columns]
    groups = df[group_column]

    print('Performing group-based train-test split')
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    identifiers_train, identifiers_test = identifiers.iloc[train_idx], identifiers.iloc[test_idx]

    # Log group distribution in training and test sets
    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    print(f"Groups in training set: {groups_train.unique()}")
    print(f"Groups in test set: {groups_test.unique()}")

    print('Normalizing data (scaling)')
    scaler = StandardScaler()
    # Combining numeric and non-numeric columns after scaling
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Apply scaling only to numeric columns
    numeric_columns = X_train.select_dtypes(include=['number']).columns
    X_train_scaled[numeric_columns] = scaler.fit_transform(X_train_scaled[numeric_columns])
    X_test_scaled[numeric_columns] = scaler.transform(X_test_scaled[numeric_columns])

    print('Transformation complete')

    # Return the scaled and unscaled data, along with the train and test indices and group information
    return (X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test,
            identifiers_train, identifiers_test, groups_train, groups_test, feature_names)

