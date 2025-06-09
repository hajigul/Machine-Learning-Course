# main.py

import os
import warnings
warnings.filterwarnings("ignore")

from data_import import load_data, preprocess_data
from model import train_random_forest, evaluate_model
from evaluate import plot_feature_importance, plot_roc_auc

RESULT_DIR = "results"

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(rf_model, X_train, y_train, X_test, y_test, output_path=os.path.join(RESULT_DIR, "metrics.txt"))

    print("Plotting feature importance...")
    plot_feature_importance(rf_model, X_train, y_train, save_path=os.path.join(RESULT_DIR, "feature_importance.png"))

    print("Plotting ROC AUC...")
    plot_roc_auc(rf_model, X_test, y_test, save_path=os.path.join(RESULT_DIR, "roc_auc.png"))

if __name__ == "__main__":
    main()