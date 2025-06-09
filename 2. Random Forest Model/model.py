# model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=5):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=0,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test, output_path="results/metrics.txt"):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"Training Accuracy: {train_acc:.3f}")
    print(f"Testing Accuracy: {test_acc:.3f}")

    report = classification_report(y_test, test_pred, output_dict=False)
    print("Classification Report:\n", report)

    # Save metrics
    with open(output_path, "w") as f:
        f.write(f"Training Accuracy: {train_acc:.3f}\n")
        f.write(f"Testing Accuracy: {test_acc:.3f}\n")
        f.write("Classification Report:\n")
        f.write(report)

    return model