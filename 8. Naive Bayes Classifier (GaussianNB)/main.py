# main.py

import os
import warnings
warnings.filterwarnings("ignore")

from data_import import load_data, preprocess_data
from model import train_naive_bayes, evaluate_model
from evaluate import plot_feature_importance, plot_roc_auc

RESULT_DIR = "results"

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training Naive Bayes Model...")
    nb_model = train_naive_bayes(X_train, y_train)

    print("Evaluating Naive Bayes Model...")
    evaluate_model(nb_model, X_train, y_train, X_test, y_test,
                   output_path=os.path.join(RESULT_DIR, "metrics_nb.txt"))

    print("Plotting feature importance...")
    try:
        plot_feature_importance(nb_model, X_train, y_train,
                                save_path=os.path.join(RESULT_DIR, "feature_importance_nb.png"))
    except Exception as e:
        print("Feature importance not available for Naive Bayes:", e)

    print("Plotting ROC AUC...")
    plot_roc_auc(nb_model, X_test, y_test,
                 save_path=os.path.join(RESULT_DIR, "roc_auc_nb.png"))

if __name__ == "__main__":
    main()