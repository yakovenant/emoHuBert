import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, RocCurveDisplay, ConfusionMatrixDisplay)


def compute_metrics(eval_pred):

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    recall = recall_score(y_true=labels, y_pred=predictions)
    precision = precision_score(y_true=labels, y_pred=predictions)
    f1 = f1_score(y_true=labels, y_pred=predictions)

    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1
    }

def plot_metrics(y_true, y_pred):
    plot_roc_auc(y_true, y_pred)
    plot_confusion(y_true, y_pred)

def plot_roc_auc(y_true, y_pred):
    RocCurveDisplay.from_predictions(y_true, y_pred)
    plt.show()

def plot_confusion(y_true, y_pred):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.show()


if __name__ == 'main':
    print('This is the metrics script.')
