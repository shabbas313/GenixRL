# -*- coding: utf-8 -*-
"""
Compare GenixRL model with other tools on an independent dataset.
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
)
from genixrl.config.config import (
    MODEL_COLORS, MODEL_INFO, MODEL_NAMES, SCORE_COLUMNS, MODEL_TO_COLUMN_IDX,
    MODEL_THRESHOLDS, DEFAULT_THRESHOLD
)
from genixrl.data.preprocessing import load_data
from genixrl.models.submodels import predict_sub_models
from genixrl.evaluation.metrics import compute_metrics
from genixrl.utils.helpers import report_class_distribution
from genixrl.visualization.plots import plot_roc_curves, plot_pr_curves

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate GenixRL on an independent dataset.")
    parser.add_argument("--data-path", default="data/ClinVar_2025_0_1.csv", help="Path to independent dataset CSV.")
    parser.add_argument("--output-dir", default="data/Eval_Results", help="Output directory for results and plots.")
    parser.add_argument("--timestamp", default="1758104492", help="Timestamp of trained model files.")
    parser.add_argument("--median-file", default=None,
                        help="Path to median file for imputation. data/outputs/all_tool_training_medians_<timestamp>.csv.")
    args = parser.parse_args()
    # Set default median_file using timestamp if not provided
    if args.median_file is None:
        args.median_file = f"data/outputs/all_tool_training_medians_{args.timestamp}.csv"
    return args

def impute_data(df, score_columns, tool_columns, median_file):
    """Impute missing values using training medians."""
    print("\nPerforming data imputation on independent dataset...")
    try:
        imputation_medians = pd.read_csv(median_file, index_col=0, header=None).squeeze("columns")
        print(f"Loaded comprehensive training data medians from '{median_file}'.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Median file '{median_file}' not found.")

    columns_to_impute = list(set(score_columns + tool_columns))
    for col in columns_to_impute:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                if col in imputation_medians.index:
                    median_val = imputation_medians[col]
                    df[col].fillna(median_val, inplace=True)
                    print(f"Imputed {missing_count} missing values in '{col}' using training median ({median_val:.4f}).")
                else:
                    print(f"Warning: No training median for '{col}'. Filling with 0.5.")
                    df[col].fillna(0.5, inplace=True)
        else:
            print(f"Warning: Column '{col}' not found in the independent dataset.")
    
    if df[score_columns + tool_columns].isnull().values.any():
        print("Missing values remain after imputation:")
        print(df[score_columns + tool_columns].isnull().sum())
        raise ValueError("NaN values remain in feature columns after imputation.")
    
    return df

def evaluate_model(y_true, y_pred, model_name, threshold, is_binary=False):
    """Evaluate a model with specified threshold."""
    result = {"Model": model_name, "Threshold": threshold}
    if not is_binary and len(np.unique(y_true)) > 1:
        try:
            result["AUC"] = roc_auc_score(y_true, y_pred)
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            result["PR-AUC"] = auc(recall, precision)
        except ValueError as e:
            print(f"Warning: AUC/PR-AUC calculation failed for {model_name}: {e}")
            result["AUC"], result["PR-AUC"] = np.nan, np.nan
    else:
        result["AUC"], result["PR-AUC"] = np.nan, np.nan
    
    pred_labels = y_pred if is_binary else (np.array(y_pred) >= threshold).astype(int)
    result["Precision"] = precision_score(y_true, pred_labels, zero_division=0)
    result["Recall (Sensitivity)"] = recall_score(y_true, pred_labels, zero_division=0)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, pred_labels).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        result["Specificity"] = specificity
    except ValueError:
        result["Specificity"] = np.nan
    result["F1-Score"] = f1_score(y_true, pred_labels, zero_division=0)
    result["MCC"] = matthews_corrcoef(y_true, pred_labels)
    return result

def main():
    args = parse_args()
    np.random.seed(42)
    
    # To create output directory if not available.
    os.makedirs(args.output_dir, exist_ok=True)

    # Comparison tools (including sub-models)
    COMPARISON_TOOLS = {
        "BayesDel_noAF": "BayesDel_noAF_Norm",
        "BayesDel_addAF": "BayesDel_addAF_Norm",
        "MetaRNN": "MetaRNN_score",
        "ClinPred": "ClinPred_score",
        
        "REVEL": "REVEL",
        "AlphaMissense": "AlphaMissense_score",
        "MVP": "MVP_score",
        "MetaSVM": "MetaSVM_Norm",
        "ESM1b": "ESM1b_norm",
        "PrimateAI": "PrimateAI_score",
        
        # -- Other Tools -- Need to un-comment for comparasion
        #"MetaLR": "MetaLR_score",
        #"Eigen": "Eigen-phred_coding_Norm",
        #"DEOGEN2": "DEOGEN2_score",
        #"MutFormer": "MutFormer_score",
        #"CADD": "CADD_PHRED_Norm",
        #"MutationAssessor": "MutationAssessor_Norm",
        #"PROVEAN": "PROVEAN_Norm",
        #"Eigen-PC": "Eigen-PC-phred_coding_Norm",
        #"gMVP": "gMVP_score",
        #"FATHMM-XF": "fathmm-XF_coding_score",
        #"EVE": "EVE_SCORE",
        #"PolyPhen": "PolyPhen",
        #"phyloP100way_vertebrate": "phyloP100way_vertebrate_Norm",
        #"LIST-S2": "LIST-S2_score",
        #"DANN": "DANN_score",
        #"SIFT": "SIFT_Invert",
        #"GERP++": "GERP++_RS_Norm",
        #"MPC": "MPC_Norm",
        #"phastCons100way": "phastCons100way_vertebrate",
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
    tool_is_binary = {tool: False for tool in COMPARISON_TOOLS}

    # Load and preprocess data
    df_independent = load_data(args.data_path, SCORE_COLUMNS + tool_columns)
    df_independent = impute_data(df_independent, SCORE_COLUMNS, tool_columns, args.median_file)
    X_independent = df_independent[SCORE_COLUMNS]
    y_independent = df_independent["clinvar_label"]
    
    if not np.all(np.isin(y_independent, [0, 1])):
        raise ValueError("clinvar_label must contain only binary labels (0 or 1).")
    
    tool_scores = {tool: df_independent[COMPARISON_TOOLS[tool]] for tool in COMPARISON_TOOLS}
    report_class_distribution(y_independent, "Independent Dataset")

    # Load saved model components
    try:
        with open(os.path.join("data/outputs", f"scaler_{args.timestamp}.pkl"), "rb") as f:
            scaler = pickle.load(f)
        print("Scaler loaded.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Scaler file 'scaler_{args.timestamp}.pkl' not found.")

    try:
        with open(os.path.join("data/outputs", f"final_weights_lr_{args.timestamp}.pkl"), "rb") as f:
            final_weights = pickle.load(f)
        print("Final weights loaded.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Final weights file 'final_weights_lr_{args.timestamp}.pkl' not found.")

    try:
        with open(os.path.join("data/outputs", f"optimal_threshold_{args.timestamp}.pkl"), "rb") as f:
            optimal_threshold = pickle.load(f)
        print(f"Optimal threshold loaded: {optimal_threshold:.4f}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Optimal threshold file 'optimal_threshold_{args.timestamp}.pkl' not found.")

    try:
        with open(os.path.join("data/outputs", f"default_threshold_{args.timestamp}.pkl"), "rb") as f:
            default_threshold = pickle.load(f)
        print(f"Default threshold loaded: {default_threshold:.4f}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Default threshold file 'default_threshold_{args.timestamp}.pkl' not found.")

    sub_models = {}
    for model_name in MODEL_NAMES:
        try:
            with open(os.path.join("data/outputs", f"sub_model_{model_name}_{args.timestamp}.pkl"), "rb") as f:
                sub_models[model_name] = pickle.load(f)
            print(f"Sub-model {model_name} loaded.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Sub-model file 'sub_model_{model_name}_{args.timestamp}.pkl' not found.")

    # Scale data
    X_independent_scaled = scaler.transform(X_independent)
    if np.any(np.isnan(X_independent_scaled)):
        raise ValueError("NaN values found in X_independent_scaled after scaling.")

    # Generate predictions
    preds_independent = predict_sub_models(sub_models, X_independent_scaled, MODEL_NAMES, MODEL_TO_COLUMN_IDX)
    for name, pred in preds_independent.items():
        if np.any(np.isnan(pred)):
            raise ValueError(f"NaN values found in predictions for {name}.")
    
    Genix_pred = sum(w * preds_independent[name] for w, name in zip(final_weights, MODEL_NAMES))
    if np.any(np.isnan(Genix_pred)):
        raise ValueError("NaN values found in GenixRL predictions.")

    # Evaluate models
    results = [
        evaluate_model(y_independent, Genix_pred, "GenixRL (Default Threshold)", default_threshold),
        evaluate_model(y_independent, Genix_pred, "GenixRL (Optimal Threshold)", optimal_threshold),
    ]
    for tool in COMPARISON_TOOLS:
        results.append(
            evaluate_model(
                y_independent, tool_scores[tool], tool, tool_thresholds.get(tool, 0.5), tool_is_binary[tool]
            )
        )

    # Save results
    results_df = pd.DataFrame(results)
    print("\nPerformance Comparison on Independent Dataset:")
    print(results_df)
    results_df.to_csv(os.path.join(args.output_dir, "comparison_results.csv"), index=False)
    print(f"Comparison results saved to '{os.path.join(args.output_dir, 'comparison_results.csv')}'")

    # Plot ROC and PR curves
    plot_roc_curves(
        y_independent, tool_scores, list(COMPARISON_TOOLS.keys()), Genix_pred,
        roc_auc_score(y_independent, Genix_pred), MODEL_COLORS, args.timestamp,
        output_dir=args.output_dir, dataset="Independent"
    )
    precision_Genix, recall_Genix, _ = precision_recall_curve(y_independent, Genix_pred)
    pr_auc_genix = auc(recall_Genix, precision_Genix)
    plot_pr_curves(
        y_independent, tool_scores, list(COMPARISON_TOOLS.keys()), Genix_pred,
        pr_auc_genix, MODEL_COLORS, args.timestamp, output_dir=args.output_dir, dataset="Independent"
    )

    # Save predictions
    output_df = df_independent.copy()
    output_df["GenixRL_Pred"] = Genix_pred
    output_df["GenixRL_Label_Default"] = [
        "Pathogenic" if p >= default_threshold else "Benign" for p in Genix_pred
    ]
    output_df["GenixRL_Label_Optimal"] = [
        "Pathogenic" if p >= optimal_threshold else "Benign" for p in Genix_pred
    ]
    for tool in COMPARISON_TOOLS:
        output_df[f"{tool}_Label"] = [
            "Pathogenic" if p >= tool_thresholds.get(tool, 0.5) else "Benign" for p in tool_scores[tool]
        ]
    output_df.to_csv(os.path.join(args.output_dir, "independent_predictions.csv"), index=False)
    print(f"Predictions saved to '{os.path.join(args.output_dir, 'independent_predictions.csv')}'")

if __name__ == "__main__":
    main()