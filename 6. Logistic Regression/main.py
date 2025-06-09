# main.py

import os
import warnings
warnings.filterwarnings("ignore")

from data_import import load_data, preprocess_data
from model import train_logistic_regression, evaluate_model
from evaluate import plot_feature_importance, plot_roc_auc

RESULT_DIR = "results"

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training Logistic Regression Model...")
    lr_model = train_logistic_regression(X_train, y_train, C=1.0, solver='liblinear', max_iter=100)

    print("Evaluating Logistic Regression Model...")
    evaluate_model(lr_model, X_train, y_train, X_test, y_test,
                   output_path=os.path.join(RESULT_DIR, "metrics_lr.txt"))

    print("Plotting feature importance...")
    try:
        plot_feature_importance(lr_model, X_train, y_train,
                                save_path=os.path.join(RESULT_DIR, "feature_importance_lr.png"))
    except Exception as e:
        print("Feature importance not available or unsupported:", e)

    print("Plotting ROC AUC...")
    plot_roc_auc(lr_model, X_test, y_test,
                 save_path=os.path.join(RESULT_DIR, "roc_auc_lr.png"))

if __name__ == "__main__":
    main()