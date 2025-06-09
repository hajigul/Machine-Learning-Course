# evaluate.py

from yellowbrick.classifier import ROCAUC
from yellowbrick.model_selection import FeatureImportances
import matplotlib.pyplot as plt
import os

def plot_roc_auc(model, X_test, y_test, classes=['stayed', 'quit'], save_path="results/roc_auc.png"):
    visualizer = ROCAUC(model, classes=classes)
    visualizer.score(X_test, y_test)
    visualizer.show(outpath=save_path)
    plt.close()

def plot_feature_importance(model, X_train, y_train, save_path="results/feature_importance.png"):
    viz = FeatureImportances(model)
    viz.fit(X_train, y_train)
    viz.show(outpath=save_path)
    plt.close()