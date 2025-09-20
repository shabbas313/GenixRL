"""
Configuration-setting.
"""

MODEL_COLORS = {
    "GenixRL": "#c70e2a",
    "BaseLine Avg.": "#888888",
    "ClinPred": "#56B4E9",
    "BayesDel_addAF": "#66C2A5",
    "MetaRNN": "#E8817D",
    "BayesDel_noAF": "#D9E257",
    "Treat All": "#555555",
    "Treat None": "#BBBBBB",
    
    # === Top-Tier Predictors 
    "REVEL": "#E377C2",
    "AlphaMissense": "#2CA02C",
    "MVP": "#9467BD",
    "MetaSVM": "#8C564B",
    "ESM1b": "#AEC7E8",
    "PrimateAI": "#008080",        
    
    # === Other Widely-Used Tools
    "SIFT": "#FF5733",   
    "PolyPhen": "#FF6F61",      
    "EVE": "#D81B60",           
    "MetaLR": "#7B1FA2",        
    "DANN": "#9CCC65",           
    "DEOGEN2": "#F06292",        
    "CADD": "#4CAF50",           
    "Eigen": "#00ACC1",          
    "Eigen-PC": "#4DD0E1",       
    "MutFormer": "#AB47BC",      
    
    # === Conservation & Other Tools Scores
    "MPC": "#820e46",            
    "PROVEAN": "#FFCA28",        
    "MutationAssessor": "#B0BEC5", 
    "FATHMM-XF": "#165c61",     
    "gMVP": "#90CAF9",           
    "LIST-S2": "#827c07",        
    "GERP++": "#8D6E63",      
    "phastCons100way": "#A5D6A7", 
    "phyloP100way_vertebrate": "#FFE082" 
}

MODEL_INFO = {
    "BayesDel_noAF": "BayesDel_noAF_Norm",
    "BayesDel_addAF": "BayesDel_addAF_Norm",
    "ClinPred": "ClinPred_score",
    "MetaRNN": "MetaRNN_score",
}

MODEL_THRESHOLDS = {
    "BayesDel_addAF": 0.664,
    "BayesDel_noAF": 0.603,
    "ClinPred": 0.5,
    "MetaRNN": 0.5,
}
DEFAULT_THRESHOLD = 0.5

# Hyperparameters
MIN_WEIGHT = 0.01
MAX_WEIGHT = 0.5
ALPHA = 0.01
GAMMA = 0.95
EPSILON = 0.1
MAX_EPISODES = 5000
BINS = 10
WEIGHT_SUM_TOLERANCE = 1e-6
METRIC_WEIGHTS = {"auc": 0.3, "pr_auc": 0.3, "mcc": 0.4}
BASELINE_REWARD = 0.01
OUTER_FOLDS = 5
INNER_FOLDS = 3
DEFAULT_THRESHOLD = 0.5

# Grid search parameters
PARAM_GRID = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l2"],
    "solver": ["lbfgs"],
}

# Derived configurations
MODEL_NAMES = list(MODEL_INFO.keys())
WEIGHT_ORDER = ["ClinPred", "BayesDel_addAF", "MetaRNN", "BayesDel_noAF"]
SCORE_COLUMNS = list(MODEL_INFO.values())
COLUMN_TO_MODEL = {v: k for k, v in MODEL_INFO.items()}
MODEL_TO_COLUMN_IDX = {model: SCORE_COLUMNS.index(col) for col, model in COLUMN_TO_MODEL.items()}