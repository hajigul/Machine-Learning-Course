# main.py

import os
import warnings
warnings.filterwarnings("ignore")

from data_import import load_data, preprocess_data
from model import train_mlp_classifier, evaluate_model
from evaluate import plot_feature_importance, plot_roc_auc

RESULT_DIR = "results"

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training MLP Classifier Model...")
    mlp_model = train_mlp_classifier(X_train, y_train, hidden_layer_sizes=(64, 32), max_iter=500)

    print("Evaluating MLP Classifier Model...")
    evaluate_model(mlp_model, X_train, y_train, X_test, y_test,
                   output_path=os.path.join(RESULT_DIR, "metrics_mlp.txt"))

    print("Plotting feature importance...")
    try:
        plot_feature_importance(mlp_model, X_train, y_train,
                                save_path=os.path.join(RESULT_DIR, "feature_importance_mlp.png"))
    except Exception as e:
        print("Feature importance not available for MLP:", e)

    print("Plotting ROC AUC...")
    plot_roc_auc(mlp_model, X_test, y_test,
                 save_path=os.path.join(RESULT_DIR, "roc_auc_mlp.png"))

if __name__ == "__main__":
    main()