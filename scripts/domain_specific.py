# -*- coding: utf-8 -*-
"""
Compare GenixRL Model with other tools on an independent dataset and optimize the decision threshold for maximum F1-score.
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    precision_score, recall_score, f1_score, matthews_corrcoef
)
from genixrl.config.config import MODEL_COLORS, MODEL_INFO, MODEL_NAMES, SCORE_COLUMNS, MODEL_TO_COLUMN_IDX, MODEL_THRESHOLDS

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Optimize GenixRL threshold and compare with other tools on an independent dataset.")
    parser.add_argument("--data-path", default="data/ClinVar_2025_0_1.csv", help="Path to independent dataset CSV.")
    parser.add_argument("--output-dir", default="data/Domain_Opt_Results", help="Output directory for results and plots.")
    parser.add_argument("--timestamp", default="1758104492", help="Timestamp of trained model files.")
    parser.add_argument("--median-file", default=None,
                        help="Path to median file for imputation. data/outputs/all_tool_training_medians_<timestamp>.csv.")
    args = parser.parse_args()
    # Set default median_file using timestamp if not provided
    if args.median_file is None:
        args.median_file = f"data/outputs/all_tool_training_medians_{args.timestamp}.csv"
    return args
    return parser.parse_args()

def load_data(data_path, required_columns):
    """Load and validate dataset."""
    try:
        df = pd.read_csv(data_path)
        dataset_name = os.path.basename(data_path).replace('.csv', '')
        print(f"Independent dataset '{dataset_name}' loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        raise FileNotFoundError(f"Independent dataset not found at {data_path}.")
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in dataset: {missing_cols}")
    
    print("Missing values in dataset:\n", df[required_columns].isnull().sum())
    return df, dataset_name

def impute_data(df, score_columns, tool_columns, median_file, timestamp):
    """Impute missing values using training medians for sub-models and dataset medians for other tools."""
    print("\nPerforming data imputation on independent dataset...")
    # Load training medians for all columns
    try:
        medians_path = median_file or f"data/outputs/all_tool_training_medians_{timestamp}.csv"
        imputation_medians = pd.read_csv(medians_path, index_col=0, header=None).squeeze("columns")
        print(f"Loaded training medians from '{medians_path}'.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Median file '{medians_path}' not found.")

    # Impute sub-model columns
    for col in score_columns:
        if col in df.columns and df[col].isnull().any():
            if col in imputation_medians.index:
                median_val = imputation_medians[col]
                df[col] = df[col].fillna(median_val)
                print(f"Imputed {df[col].isnull().sum()} missing values in '{col}' with training median ({median_val:.4f}).")
            else:
                print(f"Warning: No training median for '{col}'. Filling with 0.5.")
                df[col] = df[col].fillna(0.5)

    # Impute comparison tool columns with dataset medians
    for col in tool_columns:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Imputed {df[col].isnull().sum()} missing values in '{col}' with dataset median ({median_val:.4f}).")

    if df[score_columns + tool_columns].isnull().values.any():
        raise ValueError("NaN values remain in feature columns after imputation:\n" + str(df[score_columns + tool_columns].isnull().sum()))
    
    return df

def predict_sub_models(sub_models, X_scaled, model_names, model_to_column_idx):
    """Generate predictions from sub-models."""
    preds = {
        name: model.predict_proba(X_scaled[:, [model_to_column_idx[name]]])[:, 1]
        for name, model in sub_models.items()
    }
    return preds

def evaluate_model(y_true, y_pred, model_name, threshold):
    """Evaluate model performance."""
    result = {"Model": model_name, "Threshold": threshold}
    if len(np.unique(y_true)) > 1:
        result["AUC"] = roc_auc_score(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        result["PR-AUC"] = auc(recall, precision)
    else:
        result["AUC"], result["PR-AUC"] = np.nan, np.nan
    pred_labels = (np.array(y_pred) >= threshold).astype(int)
    result["Precision"] = precision_score(y_true, pred_labels, zero_division=0)
    result["Recall"] = recall_score(y_true, pred_labels, zero_division=0)
    result["F1-Score"] = f1_score(y_true, pred_labels, zero_division=0)
    result["MCC"] = matthews_corrcoef(y_true, pred_labels)
    return result

def plot_metrics_vs_threshold(y_true, y_pred, optimal_threshold, dataset_name, output_dir, training_optimal_threshold):
    """Plot F1-score, precision, and recall vs. threshold."""
    print("\nGenerating F1, Precision, and Recall vs. Threshold plot...")
    thresholds = np.linspace(0.01, 0.99, 500)
    f1_scores = [f1_score(y_true, (y_pred >= t).astype(int), zero_division=0) for t in thresholds]
    precision_scores = [precision_score(y_true, (y_pred >= t).astype(int), zero_division=0) for t in thresholds]
    recall_scores = [recall_score(y_true, (y_pred >= t).astype(int), zero_division=0) for t in thresholds]

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold_f1 = thresholds[optimal_idx]
    max_f1_score = f1_scores[optimal_idx]
    precision_at_max_f1 = precision_scores[optimal_idx]
    recall_at_max_f1 = recall_scores[optimal_idx]

    plt.figure(figsize=(8, 7))
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('ggplot')
    plt.plot(thresholds, f1_scores, color='#c70e2a', linewidth=3, label=f'F1-Score (Max: {max_f1_score:.3f})')
    plt.plot(thresholds, precision_scores, color='#0072B2', linestyle='--', linewidth=2, label='Precision')
    plt.plot(thresholds, recall_scores, color='#009E73', linestyle=':', linewidth=2, label='Recall')
    plt.axvline(x=optimal_threshold_f1, color='black', linestyle='-.', linewidth=1.5,
                label=f'Optimal F1 Threshold ({optimal_threshold_f1:.3f})')
    original_f1 = f1_score(y_true, (y_pred >= training_optimal_threshold).astype(int), zero_division=0)
    plt.axvline(x=training_optimal_threshold, color='purple', linestyle=':', linewidth=2,
                label=f'Training Optimal Thresh ({training_optimal_threshold:.3f})')
    plt.scatter(training_optimal_threshold, original_f1, color='purple', marker='D', s=80, zorder=10)
    annotation_text = (f'Max F1-Score: {max_f1_score:.3f}\n'
                       f'Precision: {precision_at_max_f1:.3f}\n'
                       f'Recall: {recall_at_max_f1:.3f}')
    text_x_pos = optimal_threshold_f1 + 0.2 if optimal_threshold_f1 < 0.5 else optimal_threshold_f1 - 0.2
    text_y_pos = max_f1_score - 0.2
    plt.annotate(annotation_text,
                 xy=(optimal_threshold_f1, max_f1_score),
                 xytext=(text_x_pos, text_y_pos),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, connectionstyle="arc3,rad=0.2"),
                 ha='center', va='center',
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1, alpha=0.9))
    plt.xlabel("Decision Threshold", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12, loc='lower left')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f"Metrics_vs_Threshold_{dataset_name}.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Enhanced metrics vs. threshold plot saved to '{plot_filename}'")

def plot_roc_curves(y_true, tool_scores, tool_names, genix_pred, auc_genix, model_colors, timestamp, output_dir, dataset_name):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(7, 6))
    fpr_genix, tpr_genix, _ = roc_curve(y_true, genix_pred)
    plt.plot(fpr_genix, tpr_genix, label=r"$\bf{GenixRL}$" + f" (AUC = {auc_genix:.4f})", color=model_colors["GenixRL"], linewidth=2)
    for tool in tool_names:
        if np.any(np.isnan(tool_scores[tool])):
            print(f"Warning: Skipping {tool} in ROC plot due to NaN values.")
            continue
        fpr, tpr, _ = roc_curve(y_true, tool_scores[tool])
        auc_score = roc_auc_score(y_true, tool_scores[tool])
        plt.plot(fpr, tpr, label=f"{tool} (AUC = {auc_score:.4f})", linestyle="--", color=model_colors.get(tool, '#000000'))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel("1 - Specificity (FPR)", fontsize=8)
    plt.ylabel("Sensitivity (TPR)", fontsize=8)
    plt.grid(False)
    plt.legend(loc="lower right", fontsize=6, frameon=True)
    plt.savefig(os.path.join(output_dir, f"roc_{dataset_name}_{timestamp}.png"), dpi=400, bbox_inches="tight")
    plt.close()

def plot_pr_curves(y_true, tool_scores, tool_names, genix_pred, pr_auc_genix, model_colors, timestamp, output_dir, dataset_name):
    """Plot Precision-Recall curves for all models."""
    plt.figure(figsize=(7, 6))
    precision_genix, recall_genix, _ = precision_recall_curve(y_true, genix_pred)
    plt.plot(recall_genix, precision_genix, label=r"$\bf{GenixRL}$" + f" (PR-AUC = {pr_auc_genix:.4f})", color=model_colors["GenixRL"], linewidth=2)
    for tool in tool_names:
        if np.any(np.isnan(tool_scores[tool])):
            print(f"Warning: Skipping {tool} in PR plot due to NaN values.")
            continue
        precision, recall, _ = precision_recall_curve(y_true, tool_scores[tool])
        pr_auc_score = auc(recall, precision)
        plt.plot(recall, precision, label=f"{tool} (PR-AUC = {pr_auc_score:.4f})", linestyle="--", color=model_colors.get(tool, '#000000'))
    plt.xlabel("Recall", fontsize=8)
    plt.ylabel("Precision", fontsize=8)
    plt.grid(False)
    plt.legend(loc="upper right", fontsize=6, frameon=True)
    plt.savefig(os.path.join(output_dir, f"pr_{dataset_name}_{timestamp}.png"), dpi=400, bbox_inches="tight")
    plt.close()

def main():
    args = parse_args()
    timestamp = args.timestamp
    np.random.seed(42)
    
    # To create output directory if not available
    os.makedirs(args.output_dir, exist_ok=True)

    # Comparison tools (excluding sub-models to avoid duplication in plots)
    COMPARISON_TOOLS = {
        "REVEL": "REVEL", "AlphaMissense": "AlphaMissense_score", "MVP": "MVP_score",
        "MetaSVM": "MetaSVM_Norm", "ESM1b": "ESM1b_norm", "PrimateAI": "PrimateAI_score",
        "MetaLR": "MetaLR_score", "Eigen": "Eigen-phred_coding_Norm", "DEOGEN2": "DEOGEN2_score",
        "MutFormer": "MutFormer_score", "CADD": "CADD_PHRED_Norm", "MutationAssessor": "MutationAssessor_Norm",
        "PROVEAN": "PROVEAN_Norm", "Eigen-PC": "Eigen-PC-phred_coding_Norm", "gMVP": "gMVP_score",
        "FATHMM-XF": "fathmm-XF_coding_score", "EVE": "EVE_SCORE", "PolyPhen": "PolyPhen",
        "phyloP100way_vertebrate": "phyloP100way_vertebrate_Norm", "LIST-S2": "LIST-S2_score",
        "DANN": "DANN_score", "SIFT": "SIFT_Invert", "GERP++": "GERP++_RS_Norm",
        "MPC": "MPC_Norm", "phastCons100way": "phastCons100way_vertebrate",
    }
    tool_columns = list(COMPARISON_TOOLS.values())
    tool_thresholds = {
        "ClinPred": 0.5, "BayesDel_addAF": 0.664, "MetaRNN": 0.5, "BayesDel_noAF": 0.603,
        "SIFT": 0.95, "PolyPhen": 0.9, "EVE": 0.75, "REVEL": 0.685, "AlphaMissense": 0.564,
        "DANN": 0.6, "DEOGEN2": 0.6, "ESM1b": 0.5, "Eigen-PC": 0.5, "Eigen": 0.5,
        "GERP++": 0.5, "LIST-S2": 0.75, "MPC": 0.4, "MVP": 0.75, "MetaLR": 0.5,
        "MetaSVM": 0.398, "MutFormer": 0.5, "MutationAssessor": 0.7, "PROVEAN": 0.5,
        "PrimateAI": 0.803, "FATHMM-XF": 0.5, "gMVP": 0.75, "phastCons100way": 0.5,
        "phyloP100way_vertebrate": 0.5, "CADD": 0.5,
    }

    # Load model components
    model_dir = "data/outputs"
    try:
        with open(os.path.join(model_dir, f"scaler_{timestamp}.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(model_dir, f"final_weights_lr_{timestamp}.pkl"), "rb") as f:
            final_weights = pickle.load(f)
        with open(os.path.join(model_dir, f"optimal_threshold_{timestamp}.pkl"), "rb") as f:
            optimal_threshold = pickle.load(f)
        sub_models = {}
        for name in MODEL_NAMES:
            with open(os.path.join(model_dir, f"sub_model_{name}_{timestamp}.pkl"), "rb") as f:
                sub_models[name] = pickle.load(f)
        print("All model components loaded successfully.")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"A required model file was not found: {e}")

    # Load and preprocess data
    df_independent, dataset_name = load_data(args.data_path, SCORE_COLUMNS + tool_columns + ["clinvar_label"])
    df_independent = impute_data(df_independent, SCORE_COLUMNS, tool_columns, args.median_file, timestamp)
    X_independent = df_independent[SCORE_COLUMNS]
    y_independent = df_independent["clinvar_label"]
    
    if not np.all(np.isin(y_independent, [0, 1])):
        raise ValueError("clinvar_label must contain only binary labels (0 or 1).")

    # Scale data and generate predictions
    X_independent_scaled = scaler.transform(X_independent)
    if np.any(np.isnan(X_independent_scaled)):
        raise ValueError("NaN values found in X_independent_scaled after scaling.")
    
    preds_independent = predict_sub_models(sub_models, X_independent_scaled, MODEL_NAMES, MODEL_TO_COLUMN_IDX)
    for name, pred in preds_independent.items():
        if np.any(np.isnan(pred)):
            raise ValueError(f"NaN values found in predictions for {name}.")
    
    Genix_pred = sum(w * preds_independent[name] for w, name in zip(final_weights, MODEL_NAMES))
    if np.any(np.isnan(Genix_pred)):
        raise ValueError("NaN values found in GenixRL predictions.")

    # Prepare tool scores (including sub-models for evaluation)
    tool_scores = {tool: df_independent[COMPARISON_TOOLS[tool]] for tool in COMPARISON_TOOLS}
    tool_scores.update(preds_independent)

    # Evaluate models
    results = []
    temp_thresholds = np.linspace(0.01, 0.99, 200)
    f1s = [f1_score(y_independent, (Genix_pred >= t).astype(int), zero_division=0) for t in temp_thresholds]
    empirical_optimal_thresh = temp_thresholds[np.argmax(f1s)]
    
    results.append(evaluate_model(y_independent, Genix_pred, "GenixRL (Training Optimal Thresh)", optimal_threshold))
    results.append(evaluate_model(y_independent, Genix_pred, "GenixRL (Empirical Optimal Thresh)", empirical_optimal_thresh))
    
    for tool in MODEL_NAMES + list(COMPARISON_TOOLS.keys()):
        results.append(evaluate_model(y_independent, tool_scores[tool], tool, tool_thresholds.get(tool, 0.5)))

    # Save results
    results_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False).reset_index(drop=True)
    csv_filename = os.path.join(args.output_dir, f"comparison_results_{dataset_name}.csv")
    results_df.to_csv(csv_filename, index=False)
    print(f"\nPerformance comparison results saved to '{csv_filename}'")

    # Plot metrics vs. threshold
    plot_metrics_vs_threshold(y_independent, Genix_pred, empirical_optimal_thresh, dataset_name, args.output_dir, optimal_threshold)

    # Plot ROC and PR curves
    auc_genix = roc_auc_score(y_independent, Genix_pred)
    precision_genix, recall_genix, _ = precision_recall_curve(y_independent, Genix_pred)
    pr_auc_genix = auc(recall_genix, precision_genix)
    plot_roc_curves(y_independent, tool_scores, MODEL_NAMES + list(COMPARISON_TOOLS.keys()), Genix_pred, auc_genix, MODEL_COLORS, timestamp, args.output_dir, dataset_name)
    plot_pr_curves(y_independent, tool_scores, MODEL_NAMES + list(COMPARISON_TOOLS.keys()), Genix_pred, pr_auc_genix, MODEL_COLORS, timestamp, args.output_dir, dataset_name)

    # Save predictions
    output_df = df_independent.copy()
    output_df["GenixRL_Pred"] = Genix_pred
    output_df["GenixRL_Label_Training_Optimal"] = ["Pathogenic" if p >= optimal_threshold else "Benign" for p in Genix_pred]
    output_df["GenixRL_Label_Empirical_Optimal"] = ["Pathogenic" if p >= empirical_optimal_thresh else "Benign" for p in Genix_pred]
    for tool in MODEL_NAMES + list(COMPARISON_TOOLS.keys()):
        output_df[f"{tool}_Label"] = ["Pathogenic" if p >= tool_thresholds.get(tool, 0.5) else "Benign" for p in tool_scores[tool]]
    output_df.to_csv(os.path.join(args.output_dir, f"independent_predictions_{dataset_name}.csv"), index=False)
    print(f"Predictions saved to '{os.path.join(args.output_dir, f'independent_predictions_{dataset_name}.csv')}'")
    print("\nScript finished successfully.")

if __name__ == "__main__":
    main()