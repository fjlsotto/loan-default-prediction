import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)


def evaluate_model(name, model, X_val, y_val, threshold=0.5):
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= threshold).astype(int)
    roc_auc = roc_auc_score(y_val, proba)
    avg_prec = average_precision_score(y_val, proba)

    print(f'\n=== {name} (threshold={threshold}) ===')
    print(f'ROC-AUC:       {roc_auc:.4f}')
    print(f'Avg Precision: {avg_prec:.4f}')
    print('\nClassification Report:')
    print(classification_report(y_val, preds, target_names=['No Default', 'Default']))

    return proba, roc_auc, avg_prec


def find_optimal_threshold(y_true, proba, beta=2):
    thresholds = np.arange(0.1, 0.9, 0.01)
    results = []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fb = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0
        results.append({'threshold': t, 'precision': precision, 'recall': recall, f'f{beta}': fb})
    df = pd.DataFrame(results)
    best = df.loc[df[f'f{beta}'].idxmax()]
    return best, df


def plot_roc_curves(models_dict, y_val):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (proba, color) in models_dict.items():
        fpr, tpr, _ = roc_curve(y_val, proba)
        auc = roc_auc_score(y_val, proba)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=color)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred: No Default', 'Pred: Default'],
                yticklabels=['True: No Default', 'True: Default'], ax=ax)
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    return fig
