# model.py

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def train_stacking_classifier(X_train, y_train):
    """
    Train a Stacking Classifier using:
    - Logistic Regression
    - Random Forest
    - XGBoost
    
    Final estimator: Logistic Regression

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels

    Returns:
    --------
    model : StackingClassifier
        Trained StackingClassifier model
    """
    # Define base estimators
    base_models = [
        ('lr', LogisticRegression(solver='liblinear', C=1.0, max_iter=100, random_state=0)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0, n_jobs=-1)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0))
    ]

    # Define meta-learner
    meta_model = LogisticRegression()

    # Create stacking classifier
    model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, output_path="results/metrics_stack.txt"):
    """
    Evaluate the StackingClassifier and save results.
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"Stacking Classifier Training Accuracy: {train_acc:.3f}")
    print(f"Stacking Classifier Testing Accuracy: {test_acc:.3f}")

    report = classification_report(y_test, test_pred)
    print("Classification Report:\n", report)

    # Save metrics to file
    with open(output_path, "w") as f:
        f.write(f"Training Accuracy: {train_acc:.3f}\n")
        f.write(f"Testing Accuracy: {test_acc:.3f}\n")
        f.write("Classification Report:\n")
        f.write(report)

    return model