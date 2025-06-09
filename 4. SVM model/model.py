# model.py

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os

def train_svm(X_train, y_train, kernel='rbf', C=1.0, gamma='scale'):
    
    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=0)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, output_path="results/metrics_svm.txt"):

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"SVM Training Accuracy: {train_acc:.3f}")
    print(f"SVM Testing Accuracy: {test_acc:.3f}")

    report = classification_report(y_test, test_pred)
    print("Classification Report:\n", report)

    # Save metrics to file
    with open(output_path, "w") as f:
        f.write(f"Training Accuracy: {train_acc:.3f}\n")
        f.write(f"Testing Accuracy: {test_acc:.3f}\n")
        f.write("Classification Report:\n")
        f.write(report)

    return model