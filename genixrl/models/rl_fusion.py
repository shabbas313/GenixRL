"""
Reinforcement Learning Fusion for combining sub-model predictions.
"""
import numpy as np
import random
import pickle
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, matthews_corrcoef

class ReinforcementLearningFusion:
    """
    Fuses predictions from sub-models using Q-Learning agent.
    """
    def __init__(
        self,
        models,
        model_names,
        alpha=0.01,
        gamma=0.95,
        epsilon=0.1,
        max_episodes=5000,
        bins=10,
        min_weight=0.01,
        max_weight=0.5,
        metric_weights=None,
        baseline_reward=0.01,
    ):
        self.models = models
        self.model_names = model_names
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_episodes = max_episodes
        self.bins = bins
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.metric_weights = metric_weights or {"auc": 0.3, "pr_auc": 0.3, "mcc": 0.4}
        self.baseline_reward = baseline_reward
        self.q_table = {}
        self.num_models = len(models)
        self.weights = [1 / self.num_models] * self.num_models
        self.adjustment_factors = [0.95, 1.05, 0.9, 1.1, 0.8, 1.2]
        self.actions = []
        for i in range(self.num_models):
            for factor in self.adjustment_factors:
                self.actions.append(("single", i, factor))
        for i in range(self.num_models):
            for factor in self.adjustment_factors[1::2]:
                self.actions.append(("coordinated", i, factor))
        self.actions.append(("rebalance", None, None))
        self.prev_metrics = None
        self.metric_stats = {
            "auc": {"mean": 0.5, "var": 0.01, "count": 0},
            "pr_auc": {"mean": 0.5, "var": 0.01, "count": 0},
            "mcc": {"mean": 0.0, "var": 0.01, "count": 0},
        }

    def _discretize_weights(self, weights):
        bin_edges = np.linspace(0, 1, self.bins + 1, endpoint=True)
        discretized = np.digitize(weights, bins=bin_edges, right=False)
        discretized = np.clip(discretized - 1, 0, self.bins - 1)
        return tuple(discretized)
    
    def _enforce_weight_constraints(self, weights):
        adjusted_weights = [max(self.min_weight, min(self.max_weight, w)) for w in weights]
        current_sum = sum(adjusted_weights)
        if abs(current_sum - 1.0) > 1e-6:
            adjusted_weights = [w / current_sum for w in adjusted_weights]
            adjusted_weights = [max(self.min_weight, min(self.max_weight, w)) for w in adjusted_weights]
        final_sum = sum(adjusted_weights)
        if abs(final_sum - 1.0) > 1e-6:
            adjusted_weights = [w / final_sum for w in adjusted_weights]
        return adjusted_weights

    def _apply_action(self, weights, action):
        new_weights = list(weights)
        action_type, model_idx, factor = action

        if action_type == "rebalance":
            uniform_weight = 1.0 / self.num_models
            new_weights = [w + 0.1 * (uniform_weight - w) for w in weights]
        elif action_type == "single":
            new_weights[model_idx] *= factor
        elif action_type == "coordinated":
            proposed_weight = weights[model_idx] * factor
            new_weights[model_idx] = min(proposed_weight, self.max_weight)
            remaining_weight = 1.0 - new_weights[model_idx]
            other_weights_sum = sum(w for i, w in enumerate(weights) if i != model_idx)
            if other_weights_sum > 0:
                scale = remaining_weight / other_weights_sum
                for i in range(self.num_models):
                    if i != model_idx:
                        new_weights[i] *= scale
            else:
                for i in range(self.num_models):
                    if i != model_idx:
                        new_weights[i] = remaining_weight / (self.num_models - 1)

        return self._enforce_weight_constraints(new_weights)

    def _update_metric_stats(self, metric_name, value):
        stats = self.metric_stats[metric_name]
        stats["count"] += 1
        delta = value - stats["mean"]
        stats["mean"] += delta / stats["count"]
        delta2 = value - stats["mean"]
        stats["var"] = (
            (stats["var"] * (stats["count"] - 1) + delta * delta2) / stats["count"]
            if stats["count"] > 1 else stats["var"]
        )

    def _compute_reward(self, current_metrics, prev_metrics, weights, preds_valid):
        reward = self.baseline_reward
        if prev_metrics is not None:
            for metric in ["auc", "pr_auc", "mcc"]:
                diff = current_metrics[metric] - prev_metrics[metric]
                std = np.sqrt(self.metric_stats[metric]["var"]) if self.metric_stats[metric]["var"] > 0 else 1.0
                normalized_diff = diff / max(std, 1e-6)
                reward += self.metric_weights[metric] * normalized_diff
        return reward

    def find_optimal_threshold(self, y_true, y_pred):
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        youden_j = tpr + (1 - fpr) - 1
        optimal_idx = np.argmax(youden_j)
        return thresholds[optimal_idx]

    def train(self, preds_valid, y_valid, output_dir="data/outputs", timestamp=None):
        auc_history = []
        best_auc = 0
        self.prev_metrics = None
        epsilon_decay = (self.epsilon - 0.05) / (self.max_episodes * 0.7)

        for episode in range(self.max_episodes):
            state = self._discretize_weights(self.weights)
            if state not in self.q_table:
                self.q_table[state] = {a: 0 for a in self.actions}
            if random.uniform(0, 1) > self.epsilon:
                action = max(self.q_table[state], key=self.q_table[state].get)
            else:
                action = random.choice(self.actions)

            new_weights = self._apply_action(self.weights, action)
            weighted_pred = sum(
                w * preds_valid[name]
                for w, name in zip(new_weights, self.model_names)
            )

            auc_score = roc_auc_score(y_valid, weighted_pred)
            precision, recall, _ = precision_recall_curve(y_valid, weighted_pred)
            pr_auc_score = auc(recall, precision)
            optimal_threshold = self.find_optimal_threshold(y_valid, weighted_pred)
            pred_labels = [1 if p >= optimal_threshold else 0 for p in weighted_pred]
            mcc_score = matthews_corrcoef(y_valid, pred_labels)

            current_metrics = {
                "auc": auc_score,
                "pr_auc": pr_auc_score,
                "mcc": mcc_score,
            }

            for metric, value in current_metrics.items():
                self._update_metric_stats(metric, value)

            reward = self._compute_reward(current_metrics, self.prev_metrics, new_weights, preds_valid)
            self.prev_metrics = current_metrics

            if auc_score > best_auc:
                best_auc = auc_score

            next_state = self._discretize_weights(new_weights)
            if next_state not in self.q_table:
                self.q_table[next_state] = {a: 0 for a in self.actions}
            best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
            self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
                self.alpha * (reward + self.gamma * self.q_table[next_state][best_next_action])
            self.weights = new_weights
            self.epsilon = max(0.05, self.epsilon - epsilon_decay)

            if episode % 100 == 0:
                print(
                    f"Episode {episode}: AUC = {auc_score:.4f}, PR-AUC = {pr_auc_score:.4f}, "
                    f"MCC = {mcc_score:.4f}, Threshold = {optimal_threshold:.4f}, "
                    f"Reward = {reward:.4f}, Weights = {dict(zip(self.model_names, new_weights))}"
                )
                auc_history.append((episode, auc_score))

        weights_filename = os.path.join(output_dir, f"final_weights_lr_{timestamp}.pkl")
        try:
            with open(weights_filename, "wb") as f:
                pickle.dump(self.weights, f)
            print(f"Final weights saved to {weights_filename}")
        except Exception as e:
            raise IOError(f"Failed to save weights to {weights_filename}: {e}")
        
        return self.weights, auc_history