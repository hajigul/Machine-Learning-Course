# model.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

def train_logistic_regression(X_train, y_train, C=1.0, solver='liblinear', max_iter=100):
    """
    Train a Logistic Regression classifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    C : float
        Inverse of regularization strength
    solver : str
        Algorithm to use in the optimization
    max_iter : int
        Maximum number of iterations

    Returns:
    --------
    model : LogisticRegression
        Trained Logistic Regression model
    """
    model = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=0,
        penalty='l2',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, output_path="results/metrics_lr.txt"):
    """
    Evaluate the model on training and test data and save results.
    
    Parameters:
    -----------
    model : trained classifier
    X_train : pd.DataFrame
    y_train : pd.Series
    X_test : pd.DataFrame
    y_test : pd.Series
    output_path : str
        Path to save metrics file
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"Logistic Regression Training Accuracy: {train_acc:.3f}")
    print(f"Logistic Regression Testing Accuracy: {test_acc:.3f}")

    report = classification_report(y_test, test_pred)
    print("Classification Report:\n", report)

    # Save metrics to file
    with open(output_path, "w") as f:
        f.write(f"Training Accuracy: {train_acc:.3f}\n")
        f.write(f"Testing Accuracy: {test_acc:.3f}\n")
        f.write("Classification Report:\n")
        f.write(report)

    return model