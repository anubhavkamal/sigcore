import numpy as np
from typing import Tuple


def split_train_test(dataset: Tuple[np.ndarray, np.ndarray, np.ndarray],
                     num_genuine_train: int,
                     num_genuine_test: int,
                     rng: np.random.RandomState):
    """Split dataset into per-user train (genuine only) and test (genuine + forgeries)."""
    x, y, yforg = dataset
    users = np.unique(y)

    train_idx, test_idx = [], []
    for user in users:
        genuine_idx = np.flatnonzero((y == user) & (yforg == 0))
        rng.shuffle(genuine_idx)
        user_train = genuine_idx[:num_genuine_train]
        user_test = genuine_idx[-num_genuine_test:]
        assert len(set(user_train).intersection(user_test)) == 0
        train_idx += user_train.tolist()
        test_idx += user_test.tolist()
        test_idx += np.flatnonzero((y == user) & (yforg == 1)).tolist()

    return (x[train_idx], y[train_idx], yforg[train_idx]), \
           (x[test_idx], y[test_idx], yforg[test_idx])


def build_user_training_set(user: int,
                            exp_train: Tuple[np.ndarray, np.ndarray, np.ndarray],
                            num_random_neg: int,
                            extra_negatives: np.ndarray,
                            rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """Build a binary (genuine vs random-forgery) training set for one user."""
    exp_x, exp_y, exp_yforg = exp_train

    positives = exp_x[(exp_y == user) & (exp_yforg == 0)]
    negatives_list = []

    if num_random_neg > 0:
        other_users = [u for u in np.unique(exp_y) if u != user]
        for other in other_users:
            idx = np.flatnonzero((exp_y == other) & (exp_yforg == 0))
            chosen = rng.choice(idx, num_random_neg, replace=False)
            negatives_list.append(exp_x[chosen])
        negatives_from_exp = np.concatenate(negatives_list)
    else:
        negatives_from_exp = np.empty((0, exp_x.shape[1]))

    if len(extra_negatives) > 0 and len(negatives_from_exp) > 0:
        negatives = np.concatenate([negatives_from_exp, extra_negatives])
    elif len(extra_negatives) > 0:
        negatives = extra_negatives
    elif len(negatives_from_exp) > 0:
        negatives = negatives_from_exp
    else:
        raise ValueError('No negative samples available — provide forgeries from exploitation or dev set.')

    train_x = np.concatenate([positives, negatives])
    train_y = np.concatenate([np.ones(len(positives)), np.full(len(negatives), -1)])
    return train_x, train_y


def sample_dev_negatives(dev_set: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         num_per_user: int,
                         rng: np.random.RandomState) -> np.ndarray:
    """Sample genuine signatures from a dev set to use as negative examples."""
    x, y, yforg = dev_set
    samples = []
    for user in np.unique(y):
        idx = np.flatnonzero((y == user) & (yforg == 0))
        samples.append(x[rng.choice(idx, num_per_user, replace=False)])
    return np.concatenate(samples)
