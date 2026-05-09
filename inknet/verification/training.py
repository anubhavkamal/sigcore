import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional, Dict
import sklearn.svm
import sklearn.pipeline as pipeline
import sklearn.preprocessing as preprocessing

from inknet.verification import metrics as vmetrics
from inknet.verification import data as vdata


def fit_user_classifier(training_set: Tuple[np.ndarray, np.ndarray],
                        svm_type: str,
                        C: float,
                        gamma: Optional[float]) -> sklearn.svm.SVC:
    """Train a per-user SVM classifier on (features, labels) where labels are +1/-1."""
    assert svm_type in ('linear', 'rbf')

    train_x, train_y = training_set
    n_pos = int((train_y == 1).sum())
    n_neg = int((train_y == -1).sum())
    skew = n_neg / float(n_pos) if n_pos > 0 else 1.0

    if svm_type == 'rbf':
        clf = sklearn.svm.SVC(C=C, gamma=gamma, class_weight={1: skew})
    else:
        clf = sklearn.svm.SVC(kernel='linear', C=C, class_weight={1: skew})

    pipeline_clf = pipeline.Pipeline([
        ('scaler', preprocessing.StandardScaler(with_mean=False)),
        ('classifier', clf),
    ])
    pipeline_clf.fit(train_x, train_y)
    return pipeline_clf


def evaluate_user(clf: sklearn.svm.SVC,
                  genuine: np.ndarray,
                  random_forgeries: np.ndarray,
                  skilled_forgeries: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return decision scores for genuine, random-forgery, and skilled-forgery samples."""
    return (clf.decision_function(genuine),
            clf.decision_function(random_forgeries),
            clf.decision_function(skilled_forgeries))


def fit_all_users(exp_train: Tuple[np.ndarray, np.ndarray, np.ndarray],
                  dev_set: Tuple[np.ndarray, np.ndarray, np.ndarray],
                  svm_type: str,
                  C: float,
                  gamma: float,
                  num_dev_neg: int,
                  num_exp_neg: int,
                  rng: np.random.RandomState) -> Dict[int, sklearn.svm.SVC]:
    """Train per-user classifiers for every user in the exploitation set."""
    users = np.unique(exp_train[1])
    dev_negatives = vdata.sample_dev_negatives(dev_set, num_dev_neg, rng) if num_dev_neg > 0 else []

    classifiers = {}
    for user in tqdm(users):
        train_set = vdata.build_user_training_set(user, exp_train, num_exp_neg, dev_negatives, rng)
        classifiers[user] = fit_user_classifier(train_set, svm_type, C, gamma)
    return classifiers


def score_all_users(classifiers: Dict[int, sklearn.svm.SVC],
                    exp_test: Tuple[np.ndarray, np.ndarray, np.ndarray],
                    threshold: float = 0) -> Dict:
    """Run all classifiers over the test set and compute verification metrics."""
    x, y, yforg = exp_test
    genuine_preds, random_preds, skilled_preds = [], [], []

    for user in np.unique(y):
        clf = classifiers[user]
        genuine_preds.append(clf.decision_function(x[(y == user) & (yforg == 0)]))
        skilled_preds.append(clf.decision_function(x[(y == user) & (yforg == 1)]))
        random_preds.append(clf.decision_function(x[(y != user) & (yforg == 0)]))

    all_metrics = vmetrics.compute_metrics(genuine_preds, random_preds, skilled_preds, threshold)
    print('EER (global): {:.4f}  EER (per-user): {:.4f}'.format(
        all_metrics['EER'], all_metrics['EER_userthresholds']))

    return {
        'all_metrics': all_metrics,
        'predictions': {
            'genuine': genuine_preds,
            'random': random_preds,
            'skilled': skilled_preds,
        },
    }


def run_fold(exp_set, dev_set, svm_type, C, gamma,
             num_genuine_train, num_exp_neg, num_dev_neg,
             num_genuine_test, threshold=0, rng=None):
    """Full train+test cycle for one fold."""
    if rng is None:
        rng = np.random.RandomState()
    exp_train, exp_test = vdata.split_train_test(exp_set, num_genuine_train, num_genuine_test, rng)
    classifiers = fit_all_users(exp_train, dev_set, svm_type, C, gamma, num_dev_neg, num_exp_neg, rng)
    results = score_all_users(classifiers, exp_test, threshold)
    return classifiers, results
