# model.py

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def train_mlp_classifier(X_train, y_train, hidden_layer_sizes=(64, 32), max_iter=500, alpha=0.0001):
    """
    Train a Multi-Layer Perceptron (MLP) Classifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    hidden_layer_sizes : tuple
        Number of neurons in each hidden layer
    max_iter : int
        Maximum number of training iterations
    alpha : float
        L2 regularization parameter

    Returns:
    --------
    model : MLPClassifier
        Trained MLP model
    """
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        alpha=alpha,
        random_state=0,
        early_stopping=True,
        verbose=False
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, output_path="results/metrics_mlp.txt"):
    """
    Evaluate and save performance metrics.
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"MLP Training Accuracy: {train_acc:.3f}")
    print(f"MLP Testing Accuracy: {test_acc:.3f}")

    report = classification_report(y_test, test_pred)
    print("Classification Report:\n", report)

    # Save metrics to file
    with open(output_path, "w") as f:
        f.write(f"Training Accuracy: {train_acc:.3f}\n")
        f.write(f"Testing Accuracy: {test_acc:.3f}\n")
        f.write("Classification Report:\n")
        f.write(report)

    return model