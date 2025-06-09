# model.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_decision_tree(X_train, y_train, max_depth=3, min_samples_split=2, min_samples_leaf=1):
    model = DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=0
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test, output_path="results/metrics.txt"):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"Decision Tree Training Accuracy: {train_acc:.3f}")
    print(f"Decision Tree Testing Accuracy: {test_acc:.3f}")

    # Classification report
    report = classification_report(y_test, test_pred, output_dict=False)
    print(report)

    # Save metrics to file
    with open(output_path, "w") as f:
        f.write(f"Training Accuracy: {train_acc:.3f}\n")
        f.write(f"Testing Accuracy: {test_acc:.3f}\n")
        f.write("Classification Report:\n")
        f.write(report)

    return train_pred, test_pred