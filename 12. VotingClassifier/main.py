# main.py

import os
import warnings
warnings.filterwarnings("ignore")

from data_import import load_data, preprocess_data
from model import train_voting_classifier, evaluate_model
from evaluate import plot_feature_importance, plot_roc_auc

RESULT_DIR = "results"

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training Voting Classifier Model...")
    vote_model = train_voting_classifier(X_train, y_train)

    print("Evaluating Voting Classifier Model...")
    evaluate_model(vote_model, X_train, y_train, X_test, y_test,
                   output_path=os.path.join(RESULT_DIR, "metrics_vote.txt"))

    print("Plotting feature importance...")
    try:
        plot_feature_importance(vote_model, X_train, y_train,
                                save_path=os.path.join(RESULT_DIR, "feature_importance_vote.png"))
    except Exception as e:
        print("Feature importance not available for VotingClassifier:", e)

    print("Plotting ROC AUC...")
    plot_roc_auc(vote_model, X_test, y_test,
                 save_path=os.path.join(RESULT_DIR, "roc_auc_vote.png"))

if __name__ == "__main__":
    main()