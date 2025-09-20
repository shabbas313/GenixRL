# tests/test_metrics.py
"""
Unit tests for metrics computation.
"""
import unittest
import numpy as np
from genixrl.evaluation.metrics import compute_metrics

class TestMetrics(unittest.TestCase):
    def test_compute_metrics(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0.2, 0.8, 0.7, 0.3])
        threshold = 0.5
        metrics = compute_metrics(y_true, y_pred, threshold)
        
        self.assertIn("auc", metrics)
        self.assertIn("pr_auc", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("mcc", metrics)
        self.assertEqual(metrics["Threshold"], threshold)
        self.assertGreaterEqual(metrics["auc"], 0.0)
        self.assertLessEqual(metrics["auc"], 1.0)

if __name__ == "__main__":
    unittest.main()