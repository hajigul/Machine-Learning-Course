# main.py

import os
import warnings
warnings.filterwarnings("ignore")

from data_import import load_data, preprocess_data
from model import train_adaboost, evaluate_model
from evaluate import plot_feature_importance, plot_roc_auc

RESULT_DIR = "results"

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training AdaBoost Model...")
    ada_model = train_adaboost(X_train, y_train, n_estimators=50, learning_rate=1.0)

    print("Evaluating AdaBoost Model...")
    evaluate_model(ada_model, X_train, y_train, X_test, y_test,
                   output_path=os.path.join(RESULT_DIR, "metrics_ada.txt"))

    print("Plotting feature importance...")
    plot_feature_importance(ada_model, X_train, y_train,
                            save_path=os.path.join(RESULT_DIR, "feature_importance_ada.png"))

    print("Plotting ROC AUC...")
    plot_roc_auc(ada_model, X_test, y_test,
                 save_path=os.path.join(RESULT_DIR, "roc_auc_ada.png"))

if __name__ == "__main__":
    main()