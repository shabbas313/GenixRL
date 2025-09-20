# -*- coding: utf-8 -*-
"""
GenixRL Prediction Script
Supports single variant, VCF/CSV, or pre-annotated CSV/TSV inputs.
"""
import pandas as pd
import numpy as np
import pickle
import argparse
import os
import sys
import pysam
import gzip
import myvariant
from tqdm import tqdm
import importlib.resources
from genixrl.config.config import MODEL_INFO, MODEL_NAMES, WEIGHT_ORDER, MODEL_TO_COLUMN_IDX, SCORE_COLUMNS
from genixrl.data.preprocessing import load_data
from genixrl.models.submodels import predict_sub_models

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict pathogenicity using GenixRL with automated annotation from dbNSFP.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--variant", help="A single variant to predict.\nFormats: '7:140453136:A:T'")
    group.add_argument("--input-file", help="Path to a VCF or simple CSV/TXT file with variants.\nCSV/TXT format: one variant per line as 'chrom,pos,ref,alt'")
    group.add_argument("--pre-annotated-file", help=f"Path to a CSV/TSV file with sub-model scores.\nRequired columns: BayesDel_noAF_score, BayesDel_addAF_score, ClinPred_score, MetaRNN_score")
    parser.add_argument("--output-file", default="data/predictions.csv", help="Path to save the output CSV file.")
    parser.add_argument("--dbnsfp-path", default="/home/hassan2/genixrl_scores_db.tsv.gz", help="Path to dbNSFP file for annotation.")
    parser.add_argument("--model-dir", default="data/outputs", help="Directory containing model artifacts.")
    parser.add_argument("--timestamp", default="1758104492", help="Timestamp of trained model files.")
    parser.add_argument("--median-file", default=None,
                       help="Path to median file for imputation. Defaults to data/outputs/all_tool_training_medians_<timestamp>.csv")
    parser.add_argument("--strong-threshold", type=float, default=0.709,
                       help="Threshold for Strongly Pathogenic label (default: 0.709). Scores >= this are Strongly Pathogenic.")
    args = parser.parse_args()
    # Set default median_file using timestamp if not provided
    if args.median_file is None:
        args.median_file = f"data/outputs/all_tool_training_medians_{args.timestamp}.csv"
    return args

def validate_nucleotides(ref, alt):
    """Validate that ref and alt are valid DNA nucleotides (A, T, G, C)."""
    valid_nucleotides = {'A', 'T', 'G', 'C'}
    if not (isinstance(ref, str) and isinstance(alt, str)):
        return False
    ref = ref.upper()
    alt = alt.upper()
    # Allow single nucleotides or sequences of valid nucleotides
    return all(c in valid_nucleotides for c in ref) and all(c in valid_nucleotides for c in alt)

def load_model_artifacts(model_dir, timestamp):
    """Load model artifacts (sub-models, scaler, weights, threshold)."""
    print(f"Loading model artifacts from: {model_dir}...")
    artifacts = {}
    try:
        artifacts['sub_models'] = {}
        for model_name in MODEL_NAMES:
            path = os.path.join(model_dir, f"sub_model_{model_name}_{timestamp}.pkl")
            with open(path, 'rb') as f:
                artifacts['sub_models'][model_name] = pickle.load(f)
        for key, filename in [
            ('scaler', f"scaler_{timestamp}.pkl"),
            ('final_weights', f"final_weights_lr_{timestamp}.pkl"),
            ('optimal_threshold', f"optimal_threshold_{timestamp}.pkl")
        ]:
            path = os.path.join(model_dir, filename)
            with open(path, 'rb') as f:
                artifacts[key] = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: A required model file was not found: {e}", file=sys.stderr)
        sys.exit(1)
    print("All model artifacts loaded successfully.")
    return artifacts

def scale_bayesdel(score, min_score=-1.29334, max_score=0.75731):
    """Normalize BayesDel scores to [0,1]."""
    if pd.isna(score): return np.nan
    try:
        score = float(score)
        scaled = (score - min_score) / (max_score - min_score)
        return max(0.0, min(1.0, scaled))
    except (ValueError, TypeError):
        return np.nan

def impute_data(df, score_columns, median_file):
    """Impute missing values using training medians."""
    try:
        imputation_medians = pd.read_csv(median_file, index_col=0, header=None).squeeze("columns")
    except FileNotFoundError:
        print(f"Error: Median file '{median_file}' not found.", file=sys.stderr)
        sys.exit(1)

    # Normalize BayesDel medians if raw scores are provided
    for raw_col, norm_col in [
        ('BayesDel_noAF_score', 'BayesDel_noAF_Norm'),
        ('BayesDel_addAF_score', 'BayesDel_addAF_Norm')
    ]:
        if raw_col in imputation_medians.index and norm_col not in imputation_medians.index:
            imputation_medians[norm_col] = scale_bayesdel(imputation_medians[raw_col])
            print(f"Normalized median for '{raw_col}' ({imputation_medians[raw_col]:.4f}) to '{norm_col}' ({imputation_medians[norm_col]:.4f}).")

    # Log rows with NaN before imputation
    if df[score_columns].isnull().values.any():
        nan_rows = df[df[score_columns].isnull().any(axis=1)][score_columns]
        print(f"Warning: Found {len(nan_rows)} rows with NaN values", file=sys.stderr)
        #print(nan_rows, file=sys.stderr)

    # Impute missing values
    for col in score_columns:
        if col in df.columns:
            if df[col].isnull().any():
                if col in imputation_medians.index:
                    median_val = imputation_medians[col]
                    df[col] = df[col].fillna(median_val)
                else:
                    print(f"Warning: No training median for '{col}'. Filling with 0.5.", file=sys.stderr)
                    df[col] = df[col].fillna(0.5)
        else:
            print(f"Error: Required column '{col}' not found in input data. Available columns: {', '.join(df.columns)}", file=sys.stderr)
            sys.exit(1)
    
    # Check for remaining NaN values
    if df[score_columns].isnull().values.any():
        nan_rows = df[df[score_columns].isnull().any(axis=1)][score_columns]
        print(f"Error: NaN values remain after imputation in {len(nan_rows)} rows:", file=sys.stderr)
        print(nan_rows, file=sys.stderr)
        sys.exit(1)
    
    return df

def normalize_hgvs_to_genomic(variant_str):
    """Convert HGVS notation to genomic coordinates."""
    mv = myvariant.MyVariantInfo()
    try:
        result = mv.getvariant(variant_str, fields='vcf')
        if result and 'vcf' in result:
            chrom, pos, ref, alt = result['vcf']['chr'], int(result['vcf']['pos']), result['vcf']['ref'], result['vcf']['alt']
            if not validate_nucleotides(ref, alt):
                print(f"Error: Invalid nucleotides in HGVS-converted variant '{variant_str}' (ref={ref}, alt={alt}). Must be A, T, G, C.", file=sys.stderr)
                return None, None, None, None
            return chrom, pos, ref, alt
    except Exception as e:
        print(f"Warning: Could not parse HGVS variant '{variant_str}': {e}", file=sys.stderr)
    return None, None, None, None

def parse_input_file(filepath):
    """Parse VCF or CSV/TXT file to extract variants."""
    print(f"Parsing input file: {filepath}")
    variants = []
    invalid_variants = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            line = line.strip()
            if not line: continue
            if ',' in line or len(line.split()) == 4:
                parts = line.replace(',', ' ').split()
                if len(parts) == 4:
                    try:
                        chrom, pos, ref, alt = parts[0].replace('chr',''), int(parts[1]), parts[2], parts[3]
                        if validate_nucleotides(ref, alt):
                            variants.append((chrom, pos, ref, alt))
                        else:
                            invalid_variants.append((chrom, pos, ref, alt))
                    except ValueError as e:
                        print(f"Warning: Skipping invalid variant line '{line}': {e}", file=sys.stderr)
                continue
            parts = line.split('\t')
            if len(parts) >= 5:
                try:
                    chrom, pos, _, ref, alt_alleles = parts[0:5]
                    chrom = chrom.replace('chr','')
                    pos = int(pos)
                    for alt in alt_alleles.split(','):
                        if validate_nucleotides(ref, alt):
                            variants.append((chrom, pos, ref, alt))
                        else:
                            invalid_variants.append((chrom, pos, ref, alt))
                except ValueError as e:
                    print(f"Warning: Skipping invalid VCF line '{line}': {e}", file=sys.stderr)
    
    if invalid_variants:
        print(f"XXX-Warning-XXX: Skipped {len(invalid_variants)} invalid variants with non-standard nucleotides (must be A, T, G, C):", file=sys.stderr)
        for chrom, pos, ref, alt in invalid_variants:
            print(f"  - {chrom}:{pos}:{ref}:{alt}", file=sys.stderr)
    
    print(f"Found {len(variants)} valid variants to process.")
    return variants

def annotate_variant(chrom, pos, ref, alt, dbnsfp_tabix, header_map):
    """Annotate a variant using dbNSFP."""
    scores = {'chrom': chrom, 'pos': pos, 'ref': ref, 'alt': alt}
    try:
        ref_idx = header_map['ref']
        alt_idx = header_map['alt']
        records = dbnsfp_tabix.fetch(chrom, pos - 1, pos)
        for record_line in records:
            fields = record_line.split('\t')
            if fields[ref_idx] == ref and fields[alt_idx] == alt:
                for model_key, dbnsfp_col_name in {
                    'BayesDel_noAF_score': 'BayesDel_noAF_score',
                    'BayesDel_addAF_score': 'BayesDel_addAF_score',
                    'ClinPred_score': 'ClinPred_score',
                    'MetaRNN_score': 'MetaRNN_score'
                }.items():
                    col_idx = header_map.get(dbnsfp_col_name)
                    if col_idx is not None:
                        score_str = fields[col_idx]
                        scores[model_key] = float(score_str) if score_str != '.' else np.nan
                return scores
    except (ValueError, KeyError) as e:
        print(f"Warning: Could not annotate variant {chrom}:{pos}:{ref}:{alt}: {e}", file=sys.stderr)
    # If annotation fails, return scores with NaN values for imputation
    for model_key in ['BayesDel_noAF_score', 'BayesDel_addAF_score', 'ClinPred_score', 'MetaRNN_score']:
        if model_key not in scores:
            scores[model_key] = np.nan
    return scores

def get_dbnsfp_header(dbnsfp_path):
    """Read header from dbNSFP file."""
    try:
        with gzip.open(dbnsfp_path, 'rt') as f:
            header_line = f.readline().strip()
            header_columns = header_line.replace("#", "").split('\t')
        return {name: i for i, name in enumerate(header_columns)}
    except Exception as e:
        print(f"Error: Could not read header from {dbnsfp_path}: {e}", file=sys.stderr)
        sys.exit(1)

def process_annotated_scores(df):
    """Process and normalize annotated scores."""
    # Normalize BayesDel scores first
    df['BayesDel_addAF_Norm'] = df['BayesDel_addAF_score'].apply(scale_bayesdel)
    df['BayesDel_noAF_Norm'] = df['BayesDel_noAF_score'].apply(scale_bayesdel)
    
    return df

def main():
    args = parse_args()
    np.random.seed(42)
    
    PRE_ANNOTATED_COLUMNS = ['BayesDel_noAF_score', 'BayesDel_addAF_score', 'ClinPred_score', 'MetaRNN_score']
    
    df_processed = None
    if args.pre_annotated_file:
        print("--- Running in Pre-Annotated Mode ---")
        try:
            df_input = load_data(args.pre_annotated_file, PRE_ANNOTATED_COLUMNS, require_label=False)
            df_processed = process_annotated_scores(df_input)
            df_processed = impute_data(df_processed, SCORE_COLUMNS, args.median_file)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            print(f"Ensure '{args.pre_annotated_file}' contains columns: {', '.join(PRE_ANNOTATED_COLUMNS)}", file=sys.stderr)
            print("Alternatively, use --input-file with --dbnsfp-path to annotate raw variants.", file=sys.stderr)
            sys.exit(1)
    else:
        print("--- Running in Annotation Mode ---")
        if not os.path.exists(args.dbnsfp_path):
            print(f"Error: dbNSFP file not found at {args.dbnsfp_path}", file=sys.stderr)
            sys.exit(1)
        
        variants_to_process = []
        if args.variant:
            chrom, pos, ref, alt = None, None, None, None
            if ':' in args.variant and len(args.variant.split(':')) == 4:
                parts = args.variant.split(':')
                try:
                    chrom, pos, ref, alt = parts[0].replace('chr',''), int(parts[1]), parts[2], parts[3]
                    if not validate_nucleotides(ref, alt):
                        print(f"Error: Invalid nucleotides in variant '{args.variant}' (ref={ref}, alt={alt}). Must be A, T, G, C.", file=sys.stderr)
                        sys.exit(1)
                except ValueError as e:
                    print(f"Error: Invalid variant format '{args.variant}': {e}", file=sys.stderr)
                    sys.exit(1)
            else:
                chrom, pos, ref, alt = normalize_hgvs_to_genomic(args.variant)
            if not all([chrom, pos, ref, alt]):
                print("Error: Could not process single variant input.", file=sys.stderr)
                sys.exit(1)
            variants_to_process.append((chrom, pos, ref, alt))
        else:
            variants_to_process = parse_input_file(args.input_file)
        
        if not variants_to_process:
            print("Error: No valid variants found to process.", file=sys.stderr)
            sys.exit(1)
        
        print("\nAnnotating variants from dbNSFP...")
        header_map = get_dbnsfp_header(args.dbnsfp_path)
        annotated_results = []
        with pysam.TabixFile(args.dbnsfp_path) as dbnsfp_tabix:
            for chrom, pos, ref, alt in tqdm(variants_to_process, desc="Annotating"):
                scores = annotate_variant(chrom, pos, ref, alt, dbnsfp_tabix, header_map)
                annotated_results.append(scores)
        
        print(f"Annotation complete. Processed {len(annotated_results)} variants.")
        
        if not annotated_results:
            print("Error: No variants could be annotated.", file=sys.stderr)
            sys.exit(1)
        
        df_processed = pd.DataFrame(annotated_results)
        df_processed = process_annotated_scores(df_processed)
        df_processed = impute_data(df_processed, SCORE_COLUMNS, args.median_file)
    
    artifacts = load_model_artifacts(args.model_dir, args.timestamp)
    X_scaled = artifacts['scaler'].transform(df_processed[SCORE_COLUMNS])
    if np.any(np.isnan(X_scaled)):
        print("Error: NaN values found in scaled data:", file=sys.stderr)
        print(df_processed[SCORE_COLUMNS][np.any(np.isnan(X_scaled), axis=1)], file=sys.stderr)
        sys.exit(1)
    
    preds_independent = predict_sub_models(artifacts['sub_models'], X_scaled, MODEL_NAMES, MODEL_TO_COLUMN_IDX)
    for name, pred in preds_independent.items():
        if np.any(np.isnan(pred)):
            print(f"Error: NaN values found in predictions for {name}.", file=sys.stderr)
            sys.exit(1)
    
    genixrl_scores = sum(w * preds_independent[name] for w, name in zip(artifacts['final_weights'], WEIGHT_ORDER))
    if np.any(np.isnan(genixrl_scores)):
        print("Error: NaN values found in GenixRL predictions.", file=sys.stderr)
        sys.exit(1)
    
    df_processed['GenixRL_score'] = genixrl_scores
    # Apply three-tiered labeling
    df_processed['GenixRL_Prediction'] = [
        'High-Confidence Pathogenic' if p >= args.strong_threshold else
        'Pathogenic' if p >= artifacts['optimal_threshold'] else
        'Benign' for p in genixrl_scores
    ]
    
    if args.variant:
        display_cols = [c for c in df_processed.columns if c not in ['GenixRL_score', 'GenixRL_Prediction']] + ['GenixRL_score', 'GenixRL_Prediction']
        print("\n===== GenixRL Prediction Result =====")
        print(df_processed[display_cols].to_string(index=False))
    else:
        try:
            df_processed.to_csv(args.output_file, index=False)
            print(f"\n===== Success! GenixRL Predictions saved to: {args.output_file} =====")
        except Exception as e:
            print(f"Error: Could not save output file: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()