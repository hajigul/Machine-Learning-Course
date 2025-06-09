# model.py

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def train_hist_gradient_boosting(X_train, y_train, max_iter=200, max_depth=5, learning_rate=0.1):
    """
    Train a Histogram-Based Gradient Boosting Classifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    max_iter : int
        Max number of trees
    max_depth : int
        Max depth of each tree
    learning_rate : float
        Step size shrinkage

    Returns:
    --------
    model : HistGradientBoostingClassifier
        Trained model
    """
    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=0,
        early_stopping=False  # Set to True if using validation set
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, output_path="results/metrics_hgb.txt"):
    """
    Evaluate and save performance metrics.
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"HistGradientBoosting Training Accuracy: {train_acc:.3f}")
    print(f"HistGradientBoosting Testing Accuracy: {test_acc:.3f}")

    report = classification_report(y_test, test_pred)
    print("Classification Report:\n", report)

    # Save metrics to file
    with open(output_path, "w") as f:
        f.write(f"Training Accuracy: {train_acc:.3f}\n")
        f.write(f"Testing Accuracy: {test_acc:.3f}\n")
        f.write("Classification Report:\n")
        f.write(report)

    return model