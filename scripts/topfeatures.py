import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def fit_and_plot_logistic_regression(X_train, X_test, y_train, y_test, feature_names, max_iter=1000, top_features=20, solver='saga', penalty='l2', tol=0.01):
    """
    Fits a Logistic Regression model on the numeric training data and plots the top N features based on their importance.
    Now includes performance optimizations and retains numeric column selection within the function.
    
    Parameters:
    - X_train: Training feature DataFrame.
    - X_test: Test feature DataFrame.
    - y_train: Training labels.
    - y_test: Test labels.
    - feature_names: Names of the features in the original dataset.
    - max_iter: Maximum number of iterations for Logistic Regression.
    - top_features: Number of top features to display.
    - solver: Solver to use for Logistic Regression, 'saga' recommended for large datasets.
    - penalty: Penalty (regularization term) to use, 'l2' or 'none'.
    - tol: Tolerance for stopping criteria, consider increasing for faster convergence.
    """
    print('Select numeric columns only')
    numeric_columns = X_train.select_dtypes(include=['number']).columns
    X_train_numeric = X_train[numeric_columns]
    X_test_numeric = X_test[numeric_columns]

    print('Ensure feature_names only includes names of numeric columns')
    feature_names_numeric = [name for name in feature_names if name in numeric_columns]

    print('Initialize and fit the Logistic Regression model')
    log_reg = LogisticRegression(max_iter=max_iter, solver=solver, penalty=penalty, tol=tol, n_jobs=-1)
    log_reg.fit(X_train_numeric, y_train)

    print('Extract the coefficients of the model')
    coefficients = np.abs(log_reg.coef_[0])

    print('Sort the features by the absolute value of their coefficients')
    sorted_indices = np.argsort(coefficients)[::-1]

    print('Plot the top N features based on their importance')
    plt.figure(figsize=(12, 6))
    top_indices = sorted_indices[:top_features]
    plt.bar(range(top_features), coefficients[top_indices], align='center')
    plt.xticks(range(top_features), labels=np.array(feature_names_numeric)[top_indices], rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Absolute Coefficient Value')
    plt.title(f'Top {top_features} Logistic Regression Coefficients for Numeric Features')
    plt.tight_layout()
    plt.show()
