# model.py

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import os

def train_naive_bayes(X_train, y_train):
    """
    Train a Gaussian Naive Bayes classifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels

    Returns:
    --------
    model : GaussianNB
        Trained Naive Bayes model
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, output_path="results/metrics_nb.txt"):
    """
    Evaluate the model and save metrics.
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"Naive Bayes Training Accuracy: {train_acc:.3f}")
    print(f"Naive Bayes Testing Accuracy: {test_acc:.3f}")

    report = classification_report(y_test, test_pred)
    print("Classification Report:\n", report)

    # Save metrics to file
    with open(output_path, "w") as f:
        f.write(f"Training Accuracy: {train_acc:.3f}\n")
        f.write(f"Testing Accuracy: {test_acc:.3f}\n")
        f.write("Classification Report:\n")
        f.write(report)

    return model