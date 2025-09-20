"""
Helper functions.
"""
import pandas as pd

def report_class_distribution(y, set_name="Set"):
    """
    Report class distribution of a dataset.
    """
    counts = pd.Series(y).value_counts()
    total = len(y)
    print(f"\nClass distribution in {set_name}:")
    print(f"Benign (0): {counts.get(0, 0)} ({counts.get(0, 0)/total*100:.2f}%)")
    print(f"Pathogenic (1): {counts.get(1, 0)} ({counts.get(1, 0)/total*100:.2f}%)")