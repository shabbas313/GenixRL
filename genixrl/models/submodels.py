"""
Sub-model training and prediction.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import time
import pickle
import os

def train_sub_models(X_train, y_train, score_columns, model_names, model_to_column_idx, inner_cv, param_grid, output_dir="data/outputs", timestamp=None):
    """
    Train sub-models using GridSearchCV.
    """
    sub_models = {}
    for model_name in model_names:
        column_idx = model_to_column_idx[model_name]
        X_train_sub = X_train[:, [column_idx]]
        
        lr_base = LogisticRegression(class_weight="balanced", max_iter=1000)
        start_time = time.time()
        
        grid = GridSearchCV(
            lr_base, param_grid, scoring="roc_auc", cv=inner_cv, n_jobs=1
        )
        grid.fit(X_train_sub, y_train)
        
        elapsed_time = time.time() - start_time
        print(f"GridSearchCV for {model_name} took {elapsed_time:.2f} seconds")
        
        best_model = grid.best_estimator_
        if hasattr(best_model, "n_iter_") and best_model.n_iter_[0] >= best_model.max_iter:
            print(f"Warning: {model_name} did not converge after {best_model.max_iter} iterations.")
        sub_models[model_name] = best_model
        print(f"Best parameters for {model_name}: {grid.best_params_}")
        
        # Save sub-model
        os.makedirs(output_dir, exist_ok=True)
        model_filename = os.path.join(output_dir, f"sub_model_{model_name}_{timestamp}.pkl")
        try:
            with open(model_filename, "wb") as f:
                pickle.dump(best_model, f)
            print(f"Sub-model {model_name} saved to '{model_filename}'")
        except Exception as e:
            raise IOError(f"Failed to save sub-model {model_name}: {e}")
    
    return sub_models

def predict_sub_models(sub_models, X, model_names, model_to_column_idx):
    """
    Generate predictions from sub-models.
    """
    preds = {
        name: model.predict_proba(X[:, [model_to_column_idx[name]]])[:, 1]
        for name, model in sub_models.items()
    }
    return preds