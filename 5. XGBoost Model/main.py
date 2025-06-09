# main.py

import os
import warnings
warnings.filterwarnings("ignore")

from data_import import load_data, preprocess_data
from model import train_xgboost, evaluate_model
from evaluate import plot_feature_importance, plot_roc_auc

RESULT_DIR = "results"

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training XGBoost Model...")
    xgb_model = train_xgboost(X_train, y_train, n_estimators=100, max_depth=3, learning_rate=0.1)

    print("Evaluating XGBoost Model...")
    evaluate_model(xgb_model, X_train, y_train, X_test, y_test,
                   output_path=os.path.join(RESULT_DIR, "metrics_xgb.txt"))

    print("Plotting feature importance...")
    plot_feature_importance(xgb_model, X_train, y_train,
                            save_path=os.path.join(RESULT_DIR, "feature_importance_xgb.png"))

    print("Plotting ROC AUC...")
    plot_roc_auc(xgb_model, X_test, y_test,
                 save_path=os.path.join(RESULT_DIR, "roc_auc_xgb.png"))

if __name__ == "__main__":
    main()