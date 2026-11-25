"""Training-only score alignment for partial, region-level concept heads.

Illustration-wise out-of-fold predictions train the downstream representation.
These folds are WITHIN the training source family, not independent author tests.
Sigmoid calibration is fitted to training OOF scores/labels, not test labels.
It is not an assertion that probabilities are reliable on another source.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from .cunfa_concept_supervision import IGNORE


def _head(x, y):
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Each fitted attribute head needs both known polarities')
    return LogisticRegression(C=1, class_weight='balanced', solver='lbfgs',
                              random_state=0, max_iter=2000).fit(x, y)


def fit_training_concepts(x_train, y_train, illustration_ids):
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    groups = np.asarray(illustration_ids)
    if x_train.ndim != 2 or y_train.shape[0] != len(x_train) or len(groups) != len(x_train):
        raise ValueError('Mismatched training-only arrays')
    if len(set(groups)) < 3:
        raise ValueError('Need at least three complete illustration groups')
    oof = np.full(y_train.shape, np.nan, dtype='float64')
    folds = []
    for group in sorted(set(groups)):
        fit_groups = groups != group
        held = ~fit_groups
        counts = []
        for j in range(y_train.shape[1]):
            fit = fit_groups & (y_train[:, j] != IGNORE)
            model = _head(x_train[fit], y_train[fit, j])
            oof[held, j] = model.decision_function(x_train[held])
            counts.append(int(fit.sum()))
        folds.append(dict(held_illustration=str(group),
                          fitted_illustrations=sorted(set(groups[fit_groups])),
                          known_attribute_training_counts=counts,
                          no_regions_from_held_illustration_in_base_head=True))
    if not np.isfinite(oof).all():
        raise ValueError('Incomplete OOF scores')
    heads, calibrators = [], []
    prevalence = []
    for j in range(y_train.shape[1]):
        known = y_train[:, j] != IGNORE
        heads.append(_head(x_train[known], y_train[known, j]))
        calibrator = make_pipeline(StandardScaler(),
            LogisticRegression(C=1, class_weight='balanced', solver='lbfgs',
                               random_state=0, max_iter=2000))
        calibrator.fit(oof[known, j:j+1], y_train[known, j])
        calibrators.append(calibrator)
        prevalence.append(float(y_train[known, j].mean()))
    return dict(heads=heads, calibrators=calibrators, oof_decision=oof,
                folds=folds, training_prevalence=prevalence,
                oof_scope='base head excludes entire illustration; calibrator fits all training OOF references',
                oof_calibrated_scores_not_an_unbiased_internal_test=True)


def scores_from_decisions(bundle, decisions, calibrated=False):
    decisions = np.asarray(decisions)
    if calibrated:
        return np.stack([m.predict_proba(decisions[:, j:j+1])[:, 1]
                         for j, m in enumerate(bundle['calibrators'])], axis=1)
    # The same sigmoid as sklearn's binary LogisticRegression predict_proba.
    return 1 / (1 + np.exp(-np.clip(decisions, -700, 700)))


def predict_concepts(bundle, images_features, calibrated=False):
    decisions = np.stack([m.decision_function(images_features) for m in bundle['heads']], axis=1)
    return scores_from_decisions(bundle, decisions, calibrated)


def aggregate_regions(scores, region_units, requested_units):
    region_units = np.asarray(region_units)
    values = []
    for unit in requested_units:
        selected = region_units == unit
        if not selected.any():
            raise ValueError('A requested illustration has no given review windows')
        values.append(np.asarray(scores)[selected].mean(axis=0))
    return np.stack(values)


def fit_style_from_oof(oof_concepts, region_units, named_training_units, training_style_targets):
    means = aggregate_regions(oof_concepts, region_units, named_training_units)
    # All scaler statistics come from the four TRAINING named examples, never
    # the test rows or test concept references. No raw-image feature bypass.
    model = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', class_weight='balanced'))
    model.fit(means, training_style_targets)
    return model
