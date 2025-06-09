# model.py

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def train_adaboost(X_train, y_train, n_estimators=50, learning_rate=1.0):
    """
    Train an AdaBoost Classifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    n_estimators : int
        Number of weak learners
    learning_rate : float
        Weight applied to each classifier in the ensemble

    Returns:
    --------
    model : AdaBoostClassifier
        Trained AdaBoost model
    """
    model = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=0
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, output_path="results/metrics_ada.txt"):
    """
    Evaluate the model on training and test data and save results.
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"AdaBoost Training Accuracy: {train_acc:.3f}")
    print(f"AdaBoost Testing Accuracy: {test_acc:.3f}")

    report = classification_report(y_test, test_pred)
    print("Classification Report:\n", report)

    # Save metrics to file
    with open(output_path, "w") as f:
        f.write(f"Training Accuracy: {train_acc:.3f}\n")
        f.write(f"Testing Accuracy: {test_acc:.3f}\n")
        f.write("Classification Report:\n")
        f.write(report)

    return model