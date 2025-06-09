# model.py

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def train_xgboost(X_train, y_train, n_estimators=100, max_depth=3, learning_rate=0.1):
    """
    Train an XGBoost Classifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum tree depth
    learning_rate : float
        Step size shrinkage to prevent overfitting

    Returns:
    --------
    model : XGBClassifier
        Trained XGBoost model
    """
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=0
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, output_path="results/metrics_xgb.txt"):
    """
    Evaluate the model and save metrics.
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"XGBoost Training Accuracy: {train_acc:.3f}")
    print(f"XGBoost Testing Accuracy: {test_acc:.3f}")

    report = classification_report(y_test, test_pred)
    print("Classification Report:\n", report)

    # Save metrics
    with open(output_path, "w") as f:
        f.write(f"Training Accuracy: {train_acc:.3f}\n")
        f.write(f"Testing Accuracy: {test_acc:.3f}\n")
        f.write("Classification Report:\n")
        f.write(report)

    return model