"""
Visualization functions.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap
import os
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.calibration import calibration_curve

def plot_auc_history(auc_history, timestamp, output_dir="data/outputs"):
    """
    Plot AUC improvement over RL training episodes.
    """
    plt.figure(figsize=(6, 5))
    plt.plot(*zip(*auc_history))
    plt.xlabel("Episodes")
    plt.ylabel("AUC Score (RL Validation Dataset)")
    plt.title("AUC Improvement Over RL Training (Logistic Regression, Final Run)")
    plt.savefig(os.path.join(output_dir, f"AUC_Progress_{timestamp}.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_roc_curves(y_true, preds, model_names, final_pred, auc_final, model_colors, timestamp, output_dir="data/outputs", dataset="Test"):
    """
    Plot ROC curves for all models.
    """
    plt.figure(figsize=(6, 5))
    fpr_final, tpr_final, _ = roc_curve(y_true, final_pred)
    plt.plot(fpr_final, tpr_final, label=f"GenixRL (AUC = {auc_final:.4f})", linewidth=3, color=model_colors["GenixRL"])
    for name in model_names:
        fpr, tpr, _ = roc_curve(y_true, preds[name])
        roc_auc = roc_auc_score(y_true, preds[name])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", linestyle="--", color=model_colors[name])
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel("1 - Specificity (FPR)")
    plt.ylabel("Sensitivity (TPR)")
    plt.title(f"ROC-AUC [{dataset} Dataset]")
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, f"ROC_{dataset}_{timestamp}.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_pr_curves(y_true, preds, model_names, final_pred, pr_auc_final, model_colors, timestamp, output_dir="data/outputs", dataset="Test"):
    """
    Plot PR curves for all models.
    """
    plt.figure(figsize=(6, 5))
    precision_final, recall_final, _ = precision_recall_curve(y_true, final_pred)
    plt.plot(recall_final, precision_final, label=f"GenixRL (PR-AUC = {pr_auc_final:.4f})", linewidth=3, color=model_colors["GenixRL"])
    for name in model_names:
        precision, recall, _ = precision_recall_curve(y_true, preds[name])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.4f})", linestyle="--", color=model_colors[name])
    prevalence = y_true.mean()
    plt.axhline(y=prevalence, color='gray', linestyle='--', linewidth=1, label=f'No Skill (Prevalence={prevalence:.2f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR-AUC [{dataset} Dataset]")
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(output_dir, f"PR_{dataset}_{timestamp}.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_decision_curve(dca_df, model_colors, timestamp, output_dir="data/outputs"):
    """
    Plot decision curve analysis.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(dca_df["Threshold"], dca_df["Net_Benefit_RL_Fusion"], label="GenixRL", linewidth=3, color=model_colors["GenixRL"])
    plt.plot(dca_df["Threshold"], dca_df["Net_Benefit_Baseline"], label="BaseLine Avg.", linewidth=2, linestyle="-.", color=model_colors["BaseLine Avg."])
    plt.plot(dca_df["Threshold"], dca_df["Net_Benefit_MetaRNN"], label="MetaRNN", linewidth=2, linestyle="--", color=model_colors["MetaRNN"])
    plt.plot(dca_df["Threshold"], dca_df["Net_Benefit_ClinPred"], label="ClinPred", linewidth=2, linestyle="--", color=model_colors["ClinPred"])
    plt.plot(dca_df["Threshold"], dca_df["Net_Benefit_BayesDel_addAF"], label="BayesDel_addAF", linewidth=2, linestyle="--", color=model_colors["BayesDel_addAF"])
    plt.plot(dca_df["Threshold"], dca_df["Net_Benefit_Treat_All"], label="Treat All", linestyle=":", color=model_colors["Treat All"])
    plt.plot(dca_df["Threshold"], dca_df["Net_Benefit_Treat_None"], label="Treat None", linestyle=":", color=model_colors["Treat None"])
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis (Test Set)")
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(output_dir, f"DCA_Test_{timestamp}.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_shap_summary(shap_values, X_test, score_columns, timestamp, output_dir="data/outputs"):
    """
    Plot SHAP summary.
    """
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test[score_columns], feature_names=score_columns, show=False)
    plt.title("SHAP Feature Importance for GenixRL", fontsize=14, pad=15)
    plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.gca().tick_params(axis='both', which='major', labelsize=10)
    plt.gca().set_facecolor('#F9F9F9')
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Shap_Summary_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_calibration(y_test, preds, model_names, final_pred, model_colors, timestamp, output_dir="data/outputs"):
    """
    Plot calibration curves.
    """
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle=":", label="Perfectly Calibrated", color=model_colors["Treat None"])
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, final_pred, n_bins=10, strategy='uniform')
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="GenixRL", color=model_colors["GenixRL"], linewidth=3)
    for name in model_names:
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, preds[name], n_bins=10, strategy='uniform')
        plt.plot(mean_predicted_value, fraction_of_positives, "s--", label=name, color=model_colors[name])
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot (Reliability)")
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(output_dir, f"Calibration_Plot_{timestamp}.png"), dpi=300)
    plt.close()

def plot_score_distribution(y_test, final_pred, optimal_threshold, timestamp, output_dir="data/outputs"):
    """
    Plot score distribution by class.
    """
    plt.figure(figsize=(6, 5))
    sns.kdeplot(final_pred[y_test==0], label="Benign", fill=True, color="#014a14")
    sns.kdeplot(final_pred[y_test==1], label="Pathogenic", fill=True, color="#ed0707")
    plt.title("Distribution of Predicted Scores by Class")
    plt.xlabel("GenixRL Score")
    plt.ylabel("Density")
    plt.axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.2f})')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(output_dir, f"Score_Distribution_{timestamp}.png"), dpi=300)
    plt.close()

def plot_final_weights(model_names, final_weights, model_colors, timestamp, output_dir="data/outputs"):
    """
    Plot final optimized weights.
    """
    plt.figure(figsize=(7, 5))
    weights_df = pd.DataFrame({"Sub-Model": model_names, "Weight": final_weights})
    weights_df = weights_df.sort_values("Weight", ascending=False)
    bar_colors = [model_colors[model] for model in weights_df["Sub-Model"]]
    sns.barplot(x="Weight", y="Sub-Model", hue="Sub-Model", data=weights_df, palette=bar_colors, legend=False)
    plt.title("Final Optimized Weights [GenixRL Fusion]")
    plt.xlabel("Assigned Weight")
    plt.ylabel("Sub-model")
    plt.xlim(0, max(final_weights) * 1.1)
    for index, value in enumerate(weights_df['Weight']):
        plt.text(value, index, f' {value:.3f}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Final_Weights_{timestamp}.png"), dpi=300)
    plt.close()