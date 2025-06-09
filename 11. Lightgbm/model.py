# model.py

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def train_lightgbm(X_train, y_train, n_estimators=100, max_depth=3, learning_rate=0.1):
    """
    Train a LightGBM Classifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum depth of each tree
    learning_rate : float
        Step size shrinkage to prevent overfitting

    Returns:
    --------
    model : LGBMClassifier
        Trained LightGBM model
    """
    model = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=0,
        verbosity=-1  # Suppress logs
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, output_path="results/metrics_lgb.txt"):
    """
    Evaluate the model on training and test data and save results.
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"LightGBM Training Accuracy: {train_acc:.3f}")
    print(f"LightGBM Testing Accuracy: {test_acc:.3f}")

    report = classification_report(y_test, test_pred)
    print("Classification Report:\n", report)

    # Save metrics to file
    with open(output_path, "w") as f:
        f.write(f"Training Accuracy: {train_acc:.3f}\n")
        f.write(f"Testing Accuracy: {test_acc:.3f}\n")
        f.write("Classification Report:\n")
        f.write(report)

    return model