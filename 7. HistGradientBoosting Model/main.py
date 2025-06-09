# main.py

import os
import warnings
warnings.filterwarnings("ignore")

from data_import import load_data, preprocess_data
from model import train_hist_gradient_boosting, evaluate_model
from evaluate import plot_feature_importance, plot_roc_auc

RESULT_DIR = "results"

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training HistGradientBoosting Model...")
    hgb_model = train_hist_gradient_boosting(X_train, y_train, max_iter=200, max_depth=5, learning_rate=0.1)

    print("Evaluating HistGradientBoosting Model...")
    evaluate_model(hgb_model, X_train, y_train, X_test, y_test,
                   output_path=os.path.join(RESULT_DIR, "metrics_hgb.txt"))

    print("Plotting feature importance...")
    plot_feature_importance(hgb_model, X_train, y_train,
                            save_path=os.path.join(RESULT_DIR, "feature_importance_hgb.png"))

    print("Plotting ROC AUC...")
    plot_roc_auc(hgb_model, X_test, y_test,
                 save_path=os.path.join(RESULT_DIR, "roc_auc_hgb.png"))

if __name__ == "__main__":
    main()