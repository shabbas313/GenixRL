"""
Data-preprocessing.
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(data_path, score_columns, label_column="clinvar_label", require_label=True):
    """
    Load and validate dataset.
    
    Args:
        data_path (str): Path to the dataset CSV.
        score_columns (list): List of feature columns required.
        label_column (str): Name of the label column (default: 'clinvar_label').
        require_label (bool): Whether to require the label column (default: True).
    """
    try:
        df = pd.read_csv(data_path)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        raise FileNotFoundError(f"Input dataset not found at {data_path}.")
    
    missing_cols = [col for col in score_columns if col not in df.columns]
    if require_label and label_column not in df.columns:
        missing_cols.append(label_column)
    
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    
    print("Missing values in dataset:\n", df[score_columns].isnull().sum())
    return df

def impute_and_split_data(df, score_columns, test_size=0.3, random_state=42, timestamp=None, output_dir="data/outputs"):
    """
    Impute missing values and split data into train/test sets.
    """
    X = df[score_columns]
    y = df["clinvar_label"]
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    test_indices = X_test.index
    
    print("\nPerforming data imputation correctly to prevent data leakage...")
    imputation_medians = X_train_full.median()
    os.makedirs(output_dir, exist_ok=True)
    imputation_medians.to_csv(os.path.join(output_dir, f"imputation_medians_{timestamp}.csv"))
    print("Medians for imputation (calculated from training data only) saved.")
    
    # Save medians for all numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    all_medians = df[numeric_columns].median()
    all_medians.to_csv(os.path.join(output_dir, f"all_tool_training_medians_{timestamp}.csv"))
    print(f"Comprehensive medians for all numeric columns saved to {os.path.join(output_dir, f'all_tool_training_medians_{timestamp}.csv')}")
    
    X_train_full = X_train_full.fillna(imputation_medians)
    X_test = X_test.fillna(imputation_medians)
    
    print("Missing values in X_train_full after imputation:", X_train_full.isnull().sum().sum())
    print("Missing values in X_test after imputation:", X_test.isnull().sum().sum())
    
    return X_train_full, X_test, y_train_full, y_test, test_indices

def scale_data(X_train, X_test, timestamp=None, output_dir="data/outputs"):
    """
    Scale data using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    scaler_filename = os.path.join(output_dir, f"scaler_{timestamp}.pkl")
    try:
        with open(scaler_filename, "wb") as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_filename}")
    except Exception as e:
        raise IOError(f"Failed to save scaler to {scaler_filename}: {e}")
    
    return X_train_scaled, X_test_scaled, scaler