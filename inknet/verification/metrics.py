import numpy as np
import sklearn.metrics as sk_metrics
from typing import List, Tuple, Dict


def compute_metrics(genuine_preds: List[np.ndarray],
                    random_preds: List[np.ndarray],
                    skilled_preds: List[np.ndarray],
                    global_threshold: float) -> Dict:
    """Compute verification metrics from per-user prediction lists.

    Returns a dict with FRR, FAR_random, FAR_skilled, mean_AUC, EER,
    EER_userthresholds, auc_list, and global_threshold.
    """
    all_genuine = np.concatenate(genuine_preds)
    all_random = np.concatenate(random_preds)
    all_skilled = np.concatenate(skilled_preds)

    frr = 1 - np.mean(all_genuine >= global_threshold)
    far_random = 1 - np.mean(all_random < global_threshold)
    far_skilled = 1 - np.mean(all_skilled < global_threshold)

    aucs, mean_auc = _compute_aucs(genuine_preds, skilled_preds)
    eer, global_threshold = _compute_eer(all_genuine, all_skilled)
    eer_user = _compute_eer_per_user(genuine_preds, skilled_preds)

    return {
        'FRR': frr,
        'FAR_random': far_random,
        'FAR_skilled': far_skilled,
        'mean_AUC': mean_auc,
        'EER': eer,
        'EER_userthresholds': eer_user,
        'auc_list': aucs,
        'global_threshold': global_threshold,
    }


def _compute_aucs(genuine_preds: List[np.ndarray],
                  skilled_preds: List[np.ndarray]) -> Tuple[List[float], float]:
    aucs = []
    for g, s in zip(genuine_preds, skilled_preds):
        y_true = np.ones(len(g) + len(s))
        y_true[len(g):] = -1
        y_scores = np.concatenate([g, s])
        aucs.append(sk_metrics.roc_auc_score(y_true, y_scores))
    return aucs, float(np.mean(aucs))


def _compute_eer(genuine: np.ndarray, skilled: np.ndarray) -> Tuple[float, float]:
    preds = np.concatenate([genuine, skilled])
    labels = np.concatenate([np.ones_like(genuine), np.full_like(skilled, -1)])
    fpr, tpr, thresholds = sk_metrics.roc_curve(labels, preds)
    idx = np.argmin(np.abs(fpr - (1 - tpr)))
    t = thresholds[idx]
    eer = (1 - np.mean(genuine >= t) + 1 - np.mean(skilled < t)) / 2.0
    return float(eer), float(t)


def _compute_eer_per_user(genuine_preds: List[np.ndarray],
                          skilled_preds: List[np.ndarray]) -> float:
    genuine_errors, skilled_errors = [], []
    n_genuine = n_skilled = 0

    for g, s in zip(genuine_preds, skilled_preds):
        y_true = np.ones(len(g) + len(s))
        y_true[len(g):] = -1
        fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, np.concatenate([g, s]))
        t = thresholds[np.argmin(np.abs(fpr - (1 - tpr)))]
        genuine_errors.append(np.sum(g < t))
        skilled_errors.append(np.sum(s >= t))
        n_genuine += len(g)
        n_skilled += len(s)

    eer = (float(np.sum(genuine_errors)) / n_genuine +
           float(np.sum(skilled_errors)) / n_skilled) / 2.0
    return eer
