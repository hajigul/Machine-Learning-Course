# main.py

import os
import warnings
warnings.filterwarnings("ignore")

from data_import import load_data, preprocess_data
from model import train_lightgbm, evaluate_model
from evaluate import plot_feature_importance, plot_roc_auc

RESULT_DIR = "results"

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training LightGBM Model...")
    lgb_model = train_lightgbm(X_train, y_train, n_estimators=100, max_depth=3, learning_rate=0.1)

    print("Evaluating LightGBM Model...")
    evaluate_model(lgb_model, X_train, y_train, X_test, y_test,
                   output_path=os.path.join(RESULT_DIR, "metrics_lgb.txt"))

    print("Plotting feature importance...")
    plot_feature_importance(lgb_model, X_train, y_train,
                            save_path=os.path.join(RESULT_DIR, "feature_importance_lgb.png"))

    print("Plotting ROC AUC...")
    plot_roc_auc(lgb_model, X_test, y_test,
                 save_path=os.path.join(RESULT_DIR, "roc_auc_lgb.png"))

if __name__ == "__main__":
    main()