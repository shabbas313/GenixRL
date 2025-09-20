"""
Main training script.
"""
import pandas as pd
import numpy as np
import time
import pickle
import os
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, f1_score, matthews_corrcoef
from genixrl.config.config import (
    MODEL_COLORS, MODEL_INFO, MODEL_THRESHOLDS, MODEL_NAMES, SCORE_COLUMNS,
    MODEL_TO_COLUMN_IDX, MIN_WEIGHT, MAX_WEIGHT, METRIC_WEIGHTS, BASELINE_REWARD,
    OUTER_FOLDS, INNER_FOLDS, DEFAULT_THRESHOLD, PARAM_GRID
)
from genixrl.data.preprocessing import load_data, impute_and_split_data, scale_data
from genixrl.models.submodels import train_sub_models, predict_sub_models
from genixrl.models.rl_fusion import ReinforcementLearningFusion
from genixrl.evaluation.metrics import compute_metrics, evaluate_sub_models, decision_curve_analysis
from genixrl.visualization.plots import (
    plot_auc_history, plot_roc_curves, plot_pr_curves, plot_decision_curve,
    plot_shap_summary, plot_calibration, plot_score_distribution, plot_final_weights
)
from genixrl.utils.helpers import report_class_distribution
import shap

def main():
    # Set random seeds
    np.random.seed(42)
    random.seed(42)
    timestamp = int(time.time())
    output_dir = "data/outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Start timing
    start_time_total = time.time()

    # Load and preprocess data
    df = load_data("data/ClinVar_2024.csv", SCORE_COLUMNS)
    X_train_full, X_test, y_train_full, y_test, test_indices = impute_and_split_data(
        df, SCORE_COLUMNS, timestamp=timestamp, output_dir=output_dir
    )
    X_train_full_scaled, X_test_scaled, scaler = scale_data(
        X_train_full, X_test, timestamp=timestamp, output_dir=output_dir
    )
    report_class_distribution(y_test, "Test Set")

    # Nested Cross-Validation
    outer_kf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=42)
    inner_kf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=42)
    nested_results = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer_kf.split(X_train_full, y_train_full)):
        print(f"\nOuter Fold {outer_fold + 1}")
        X_outer_train = X_train_full.iloc[train_idx]
        y_outer_train = y_train_full.iloc[train_idx]
        X_outer_test = X_train_full.iloc[test_idx]
        y_outer_test = y_train_full.iloc[test_idx]

        X_subtrain, X_rl_valid, y_subtrain, y_rl_valid = train_test_split(
            X_outer_train, y_outer_train, test_size=0.2, random_state=42, stratify=y_outer_train
        )

        scaler = StandardScaler()
        X_subtrain_scaled = scaler.fit_transform(X_subtrain)
        X_rl_valid_scaled = scaler.transform(X_rl_valid)
        X_outer_test_scaled = scaler.transform(X_outer_test)

        report_class_distribution(y_subtrain, f"Outer Fold {outer_fold + 1} Sub-Training Set")
        report_class_distribution(y_rl_valid, f"Outer Fold {outer_fold + 1} RL Validation Set")
        report_class_distribution(y_outer_test, f"Outer Fold {outer_fold + 1} Test Set")

        fold_sub_models = train_sub_models(
            X_subtrain_scaled, y_subtrain, SCORE_COLUMNS, MODEL_NAMES, MODEL_TO_COLUMN_IDX,
            inner_cv=inner_kf, param_grid=PARAM_GRID, output_dir=output_dir, timestamp=timestamp
        )

        preds_rl_valid = predict_sub_models(fold_sub_models, X_rl_valid_scaled, MODEL_NAMES, MODEL_TO_COLUMN_IDX)
        preds_test_fold = predict_sub_models(fold_sub_models, X_outer_test_scaled, MODEL_NAMES, MODEL_TO_COLUMN_IDX)

        sub_model_metrics = evaluate_sub_models(preds_rl_valid, y_rl_valid, MODEL_NAMES, MODEL_THRESHOLDS)
        weak_models = [name for name, metrics in sub_model_metrics.items() if metrics["auc"] < 0.7]
        if weak_models:
            print(f"Consider pruning weak models (AUC < 0.7): {weak_models}")

        rl_model = ReinforcementLearningFusion(
            models=list(fold_sub_models.values()), model_names=MODEL_NAMES,
            min_weight=MIN_WEIGHT, max_weight=MAX_WEIGHT, metric_weights=METRIC_WEIGHTS,
            baseline_reward=BASELINE_REWARD
        )
        final_weights, auc_history = rl_model.train(preds_rl_valid, y_rl_valid, output_dir=output_dir, timestamp=timestamp)

        final_pred_valid = sum(w * preds_rl_valid[name] for w, name in zip(final_weights, MODEL_NAMES))
        optimal_threshold = rl_model.find_optimal_threshold(y_rl_valid, final_pred_valid)
        print(f"Outer Fold {outer_fold + 1} - Optimal Threshold (Youden’s J): {optimal_threshold:.4f}")

        final_pred_fold = sum(w * preds_test_fold[name] for w, name in zip(final_weights, MODEL_NAMES))

        genix_rl_metrics_default = compute_metrics(y_outer_test, final_pred_fold, DEFAULT_THRESHOLD)
        genix_rl_metrics_default["Model"] = "GenixRL (Default Threshold)"
        genix_rl_metrics_default["Outer Fold"] = outer_fold + 1

        genix_rl_metrics_optimal = compute_metrics(y_outer_test, final_pred_fold, optimal_threshold)
        genix_rl_metrics_optimal["Model"] = "GenixRL (Optimal Threshold)"
        genix_rl_metrics_optimal["Outer Fold"] = outer_fold + 1

        nested_results.append(genix_rl_metrics_default)
        nested_results.append(genix_rl_metrics_optimal)

        for name in MODEL_NAMES:
            threshold = MODEL_THRESHOLDS[name]
            sub_model_pred = preds_test_fold[name]
            sub_model_metrics = compute_metrics(y_outer_test, sub_model_pred, threshold)
            sub_model_metrics["Model"] = name
            sub_model_metrics["Outer Fold"] = outer_fold + 1
            nested_results.append(sub_model_metrics)

    # Save nested CV results
    nested_results_df = pd.DataFrame(nested_results)
    metrics = ["auc", "pr_auc", "precision", "recall", "f1", "mcc"]
    models_to_evaluate = ["GenixRL (Default Threshold)", "GenixRL (Optimal Threshold)"] + MODEL_NAMES

    print("\nNested Stratified Cross-Validation Results:")
    for model in models_to_evaluate:
        print(f"\n{model}:")
        model_results = nested_results_df[nested_results_df["Model"] == model]
        summary_data = []
        for metric in metrics:
            mean = model_results[metric].mean()
            std = model_results[metric].std()
            summary_data.append({"Metric": metric, "Mean": mean, "Std": std})
            print(f"{metric}\t{mean:.9f}\t{std:.9f}")
        
        summary_df = pd.DataFrame(summary_data)
        model_filename = model.replace(" ", "_").replace("(", "").replace(")", "")
        summary_df.to_csv(os.path.join(output_dir, f"nested_cv_{model_filename}_{timestamp}.csv"), index=False)
        print(f"Nested CV results for {model} saved to 'nested_cv_{model_filename}_{timestamp}.csv'")

    # Final Model Training
    X_final_subtrain, X_final_rl_valid, y_final_subtrain, y_final_rl_valid = train_test_split(
        X_train_full_scaled, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    report_class_distribution(y_final_subtrain, "Final Sub-Training Set")
    report_class_distribution(y_final_rl_valid, "Final RL Validation Set")

    final_sub_models = train_sub_models(
        X_final_subtrain, y_final_subtrain, SCORE_COLUMNS, MODEL_NAMES, MODEL_TO_COLUMN_IDX,
        inner_cv=inner_kf, param_grid=PARAM_GRID, output_dir=output_dir, timestamp=timestamp
    )

    preds_final_rl_valid = predict_sub_models(final_sub_models, X_final_rl_valid, MODEL_NAMES, MODEL_TO_COLUMN_IDX)
    preds_test = predict_sub_models(final_sub_models, X_test_scaled, MODEL_NAMES, MODEL_TO_COLUMN_IDX)

    sub_model_metrics = evaluate_sub_models(preds_final_rl_valid, y_final_rl_valid, MODEL_NAMES, MODEL_THRESHOLDS)
    weak_models = [name for name, metrics in sub_model_metrics.items() if metrics["auc"] < 0.7]
    if weak_models:
        print(f"Consider pruning weak models (AUC < 0.7) for final model: {weak_models}")

    preds_test_filename = os.path.join(output_dir, f"preds_test_{timestamp}.pkl")
    try:
        with open(preds_test_filename, "wb") as f:
            pickle.dump(preds_test, f)
        print(f"Test predictions saved to {preds_test_filename}")
    except Exception as e:
        raise IOError(f"Failed to save predictions to {preds_test_filename}: {e}")

    rl_model_final = ReinforcementLearningFusion(
        models=list(final_sub_models.values()), model_names=MODEL_NAMES,
        min_weight=MIN_WEIGHT, max_weight=MAX_WEIGHT, metric_weights=METRIC_WEIGHTS,
        baseline_reward=BASELINE_REWARD
    )
    final_weights, final_auc_history = rl_model_final.train(preds_final_rl_valid, y_final_rl_valid, output_dir=output_dir, timestamp=timestamp)

    print("\nFinal Optimized Weights for Each Sub-Model:")
    for name, weight in zip(MODEL_NAMES, final_weights):
        print(f"{name}: {weight:.4f}")

    weights_df = pd.DataFrame({"Sub-Model": MODEL_NAMES, "Weight": final_weights})
    weights_df.to_csv(os.path.join(output_dir, f"final_submodel_weights_{timestamp}.csv"), index=False)
    print(f"Sub-model weights saved to 'final_submodel_weights_{timestamp}.csv'")

    final_pred_test = sum(w * preds_test[name] for w, name in zip(final_weights, MODEL_NAMES))
    final_auc_test = roc_auc_score(y_test, final_pred_test)

    precision_final, recall_final, _ = precision_recall_curve(y_test, final_pred_test)
    pr_auc_final = auc(recall_final, precision_final)

    optimal_threshold = rl_model_final.find_optimal_threshold(
        y_final_rl_valid,
        sum(w * preds_final_rl_valid[name] for w, name in zip(final_weights, MODEL_NAMES))
    )
    print(f"\nOptimal Threshold (Youden’s J): {optimal_threshold:.4f}")

    try:
        with open(os.path.join(output_dir, f"optimal_threshold_{timestamp}.pkl"), "wb") as f:
            pickle.dump(optimal_threshold, f)
        print(f"Optimal threshold saved to 'optimal_threshold_{timestamp}.pkl'")
    except Exception as e:
        raise IOError(f"Failed to save optimal threshold: {e}")

    try:
        with open(os.path.join(output_dir, f"default_threshold_{timestamp}.pkl"), "wb") as f:
            pickle.dump(DEFAULT_THRESHOLD, f)
        print(f"Default threshold saved to 'default_threshold_{timestamp}.pkl'")
    except Exception as e:
        raise IOError(f"Failed to save default threshold: {e}")

    final_pred_labels_default = [1 if p >= DEFAULT_THRESHOLD else 0 for p in final_pred_test]
    final_precision_default = precision_score(y_test, final_pred_labels_default, zero_division=0)
    final_recall_default = recall_score(y_test, final_pred_labels_default, zero_division=0)
    final_f1_default = f1_score(y_test, final_pred_labels_default, zero_division=0)
    final_mcc_default = matthews_corrcoef(y_test, final_pred_labels_default)

    final_pred_labels_optimal = [1 if p >= optimal_threshold else 0 for p in final_pred_test]
    final_precision_optimal = precision_score(y_test, final_pred_labels_optimal, zero_division=0)
    final_recall_optimal = recall_score(y_test, final_pred_labels_optimal, zero_division=0)
    final_f1_optimal = f1_score(y_test, final_pred_labels_optimal, zero_division=0)
    final_mcc_optimal = matthews_corrcoef(y_test, final_pred_labels_optimal)

    print("\nFinal Test Set Results (Logistic Regression, Default Threshold 0.5):")
    print(f"AUC: {final_auc_test:.4f}")
    print(f"PR-AUC: {pr_auc_final:.4f}")
    print(f"Precision: {final_precision_default:.4f}")
    print(f"Recall: {final_recall_default:.4f}")
    print(f"F1-Score: {final_f1_default:.4f}")
    print(f"MCC: {final_mcc_default:.4f}")

    print("\nFinal Test Set Results (Logistic Regression, Optimal Threshold):")
    print(f"AUC: {final_auc_test:.4f}")
    print(f"PR-AUC: {pr_auc_final:.4f}")
    print(f"Precision: {final_precision_optimal:.4f}")
    print(f"Recall: {final_recall_optimal:.4f}")
    print(f"F1-Score: {final_f1_optimal:.4f}")
    print(f"MCC: {final_mcc_optimal:.4f}")

    # Baseline comparison
    baseline_weights = [1 / len(MODEL_NAMES)] * len(MODEL_NAMES)
    baseline_pred_test = sum(w * preds_test[name] for w, name in zip(baseline_weights, MODEL_NAMES))
    baseline_auc_test = roc_auc_score(y_test, baseline_pred_test)
    precision_base, recall_base, _ = precision_recall_curve(y_test, baseline_pred_test)
    pr_auc_base = auc(recall_base, precision_base)
    baseline_pred_labels = [1 if p >= DEFAULT_THRESHOLD else 0 for p in baseline_pred_test]
    baseline_precision = precision_score(y_test, baseline_pred_labels, zero_division=0)
    baseline_recall = recall_score(y_test, baseline_pred_labels, zero_division=0)
    baseline_f1 = f1_score(y_test, baseline_pred_labels, zero_division=0)
    baseline_mcc = matthews_corrcoef(y_test, baseline_pred_labels)

    print("\nBaseline Test Set Results (Simple Average Ensemble, Threshold 0.5):")
    print(f"AUC: {baseline_auc_test:.4f}")
    print(f"PR-AUC: {pr_auc_base:.4f}")
    print(f"Precision: {baseline_precision:.4f}")
    print(f"Recall: {baseline_recall:.4f}")
    print(f"F1-Score: {baseline_f1:.4f}")
    print(f"MCC: {baseline_mcc:.4f}")

    # Decision Curve Analysis
    print("\nPerforming Decision Curve Analysis on Test Set...")
    thresholds, nb_rl, nb_all, nb_none = decision_curve_analysis(y_test, final_pred_test)
    _, nb_baseline, _, _ = decision_curve_analysis(y_test, baseline_pred_test)
    _, nb_metarnn, _, _ = decision_curve_analysis(y_test, preds_test["MetaRNN"])
    _, nb_clinpred, _, _ = decision_curve_analysis(y_test, preds_test["ClinPred"])
    _, nb_bayesdel, _, _ = decision_curve_analysis(y_test, preds_test["BayesDel_addAF"])

    dca_df = pd.DataFrame(
        {
            "Threshold": thresholds,
            "Net_Benefit_RL_Fusion": nb_rl,
            "Net_Benefit_Baseline": nb_baseline,
            "Net_Benefit_MetaRNN": nb_metarnn,
            "Net_Benefit_ClinPred": nb_clinpred,
            "Net_Benefit_BayesDel_addAF": nb_bayesdel,
            "Net_Benefit_Treat_All": nb_all,
            "Net_Benefit_Treat_None": nb_none,
        }
    )
    dca_df.to_csv(os.path.join(output_dir, f"dca_results_{timestamp}.csv"), index=False)
    print(f"DCA results saved to 'dca_results_{timestamp}.csv'")

    # SHAP Explainability
    print("\nComputing SHAP explanations...")
    def rl_fused_predict(X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=SCORE_COLUMNS)
        X_reordered = X[SCORE_COLUMNS]
        X_scaled = scaler.transform(X_reordered)
        preds = np.array([model.predict_proba(X_scaled[:, [MODEL_TO_COLUMN_IDX[name]]])[:, 1]
                          for name, model in final_sub_models.items()]).T
        return np.dot(preds, final_weights)

    background_data = X_train_full[SCORE_COLUMNS].iloc[:100]
    explainer = shap.KernelExplainer(rl_fused_predict, background_data)
    shap_values = explainer.shap_values(X_test[SCORE_COLUMNS], nsamples=100)

    shap_df = pd.DataFrame(shap_values, columns=SCORE_COLUMNS, index=test_indices)
    shap_df.to_csv(os.path.join(output_dir, f"shap_values_{timestamp}.csv"))
    print(f"SHAP values saved to 'shap_values_{timestamp}.csv'")

    # Save test predictions
    test_df = df.loc[test_indices].copy()
    test_df["predicted_probability_fused"] = final_pred_test
    test_df["predicted_label_fused_default"] = [
        "Pathogenic" if p >= DEFAULT_THRESHOLD else "Benign" for p in final_pred_test
    ]
    test_df["predicted_label_fused_optimal"] = [
        "Pathogenic" if p >= optimal_threshold else "Benign" for p in final_pred_test
    ]
    test_df.to_csv(os.path.join(output_dir, f"test_predictions_lr_fused_{timestamp}.csv"), index=False)
    test_indices_filename = os.path.join(output_dir, f"test_indices_{timestamp}.pkl")
    try:
        with open(test_indices_filename, "wb") as f:
            pickle.dump(test_indices, f)
        print(f"Test data with predictions saved to 'test_predictions_lr_fused_{timestamp}.csv'")
    except Exception as e:
        raise IOError(f"Failed to save test indices: {e}")

    # Plotting
    final_pred_valid = sum(w * preds_final_rl_valid[name] for w, name in zip(final_weights, MODEL_NAMES))
    final_auc_valid = roc_auc_score(y_final_rl_valid, final_pred_valid)

    plot_auc_history(final_auc_history, timestamp, output_dir=output_dir)
    plot_roc_curves(y_final_rl_valid, preds_final_rl_valid, MODEL_NAMES, final_pred_valid, final_auc_valid, MODEL_COLORS, timestamp, output_dir=output_dir, dataset="Validation")
    plot_roc_curves(y_test, preds_test, MODEL_NAMES, final_pred_test, final_auc_test, MODEL_COLORS, timestamp, output_dir=output_dir, dataset="Test")
    plot_pr_curves(y_final_rl_valid, preds_final_rl_valid, MODEL_NAMES, final_pred_valid, auc(*precision_recall_curve(y_final_rl_valid, final_pred_valid)[1::-1]), MODEL_COLORS, timestamp, output_dir=output_dir, dataset="Validation")
    plot_pr_curves(y_test, preds_test, MODEL_NAMES, final_pred_test, pr_auc_final, MODEL_COLORS, timestamp, output_dir=output_dir, dataset="Test")
    plot_decision_curve(dca_df, MODEL_COLORS, timestamp, output_dir=output_dir)
    plot_shap_summary(shap_values, X_test, SCORE_COLUMNS, timestamp, output_dir=output_dir)
    plot_calibration(y_test, preds_test, MODEL_NAMES, final_pred_test, MODEL_COLORS, timestamp, output_dir=output_dir)
    plot_score_distribution(y_test, final_pred_test, optimal_threshold, timestamp, output_dir=output_dir)
    plot_final_weights(MODEL_NAMES, final_weights, MODEL_COLORS, timestamp, output_dir=output_dir)

    # Execution time
    end_time_total = time.time()
    total_time_seconds = end_time_total - start_time_total
    hours, rem = divmod(total_time_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTotal Execution Time: {total_time_seconds:.2f} seconds")
    print(f"Total Execution Time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f} (hh:mm:ss)")

if __name__ == "__main__":
    main()