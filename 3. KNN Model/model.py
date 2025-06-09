# model.py

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def train_knn(X_train, y_train, n_neighbors=5):
    """
    Train a K-Nearest Neighbors classifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    n_neighbors : int
        Number of neighbors to use (default = 5)

    Returns:
    --------
    model : KNeighborsClassifier
        Trained KNN model
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, output_path="results/metrics_knn.txt"):
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

    print(f"KNN Training Accuracy: {train_acc:.3f}")
    print(f"KNN Testing Accuracy: {test_acc:.3f}")

    report = classification_report(y_test, test_pred)
    print("Classification Report:\n", report)

    # Save metrics to file
    with open(output_path, "w") as f:
        f.write(f"Training Accuracy: {train_acc:.3f}\n")
        f.write(f"Testing Accuracy: {test_acc:.3f}\n")
        f.write("Classification Report:\n")
        f.write(report)

    return model