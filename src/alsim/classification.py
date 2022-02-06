import sklearn
import scipy
import numpy as np
from sklearn.linear_model import LogisticRegression


def vectorize_docs(vector_source, docs):
    """
    vector_source should be a key contained in the doc, with a Numpy ndarray value.
    """
    X = np.vstack([td[vector_source] for td in docs])
    return X


def downsample_majority_class(majority_class, target_proportion, X, y):
    minority_class = np.abs(majority_class - 1)
    minority_inds = np.nonzero(y == minority_class)[0]
    minority_count = len(minority_inds)

    n_majority_class_to_keep = int(
        np.ceil((minority_count / target_proportion) - minority_count)
    )
    majority_class_inds = np.nonzero(y == majority_class)[0]
    if len(majority_class_inds) == n_majority_class_to_keep:
        # no need to do any downsampling
        # this should be a rare edge case caused by the use of np.ceil
        # print(f"mistaken call to downsample {minority_class} / {minority_class} + {majority_class} = {minority_class / (minority_class + majority_class):.2f} with to_keep = {n_majority_class_to_keep}")
        return X, y
    assert len(majority_class_inds) > n_majority_class_to_keep

    majority_inds_to_keep = np.random.choice(
        majority_class_inds, size=n_majority_class_to_keep, replace=False
    )
    # print(f"Downsampled to {n_majority_class_to_keep} / {len(majority_class_inds)} majority class (label={majority_class}) documents (minority={minority_count}, downsampled minority pct ={n_majority_class_to_keep / (n_majority_class_to_keep + minority_count) *100:.2f}%, original = {len(majority_class_inds) / (len(majority_class_inds) + minority_count) *100:.2f}%).")
    inds_to_keep = np.concatenate((majority_inds_to_keep, minority_inds))
    assert len(inds_to_keep) > 0
    X = X[inds_to_keep]
    y = y[inds_to_keep]
    assert X.shape[0] == len(inds_to_keep)
    assert y.shape[0] == len(inds_to_keep)
    return X, y


def train_model(X, y, X_valid, config):
    """
    train_model uses the following config keys:
     - learner
     - use_bbsc
     - undersample_to_proportion
    """
    if config.undersample_to_proportion:
        undersample_to_proportion = config.undersample_to_proportion
        pos_count = np.sum(y)
        pos_pct = pos_count / len(y)  # what pct of the labels are 1?
        if pos_pct < undersample_to_proportion:
            # undersample until at least the target proportion is reached
            X, y = downsample_majority_class(0, undersample_to_proportion, X, y)
        elif pos_pct > (1 - undersample_to_proportion):
            # need to undersample the positive class
            X, y = downsample_majority_class(1, undersample_to_proportion, X, y)

    if config.learner == "logreg":
        clf = LogisticRegression(
            C=1.0,
            solver="liblinear",
        )
        clf.fit(X, y)
    else:
        raise ValueError(f"Unknown learner '{config.learner}'.")
    y_unlabeled_valid_pred = clf.predict(X_valid)
    unlabeled_valid_pct_pos = np.sum(y_unlabeled_valid_pred) / len(
        y_unlabeled_valid_pred
    )

    if config.use_bbsc:
        BBSC_MIN_LABELED_DATA_COUNT = 20
        BBSC_MAX_TRAIN_FOLDS = 10  # should be in the config, but this is the number of folds used for predicting positive class proportion and thus the K-S test
        if len(y) < BBSC_MIN_LABELED_DATA_COUNT:
            # BBSC is highly unstable for small confusion matrices
            # so for now we just prevent the use of BBSC when the
            # available labeled sample is very small
            return clf, unlabeled_valid_pct_pos

        # use bbsc
        y_valid_pred = np.zeros_like(y)
        y_valid_pred_proba = np.zeros_like(y, dtype=float)
        n_splits = min(BBSC_MAX_TRAIN_FOLDS, len(y))
        kf = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=False)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pos_count = np.sum(y_train)
            if pos_count == 0:
                # the fold contains no positive training examples
                # so predict negative class
                y_pred = np.zeros_like(y_test)
                y_valid_pred_proba[test_index] = 0.0
            elif pos_count == len(y_train):
                # the fold contains only positive training examples
                # so predict positive class
                y_pred = np.ones_like(y_test)
                y_valid_pred_proba[test_index] = 1.0
            else:  # 1+ pos and neg training examples
                cv_clf = LogisticRegression(C=1.0, solver="liblinear")
                cv_clf.fit(X_train, y_train)
                y_pred = cv_clf.predict(X_test)
                y_valid_pred_proba[test_index] = cv_clf.predict_proba(X_test)[:, 1]
            y_valid_pred[test_index] = y_pred

        # use black-box shift correction
        y_unlabeled_pred = clf.predict(X_valid)
        y_unlabeled_pred_proba = clf.predict_proba(X_valid)[:, 1]
        ks_result = scipy.stats.ks_2samp(y_valid_pred_proba, y_unlabeled_pred_proba)
        p = ks_result.pvalue

        source_predicted_y0 = np.sum(y_valid_pred == 0) / len(y_valid_pred)
        source_predicted_y1 = np.sum(y_valid_pred == 1) / len(y_valid_pred)

        labeled_y0 = np.sum(y == 0) / len(y)
        labeled_y1 = np.sum(y == 1) / len(y)
        v_est = np.array([labeled_y0, labeled_y1])
        # C_est is the normalized confusion matrix on the validation data
        C_est = np.zeros((2, 2))
        C_est[0, 0] = np.sum((y == 0) & (y_valid_pred == 0))
        C_est[0, 1] = np.sum((y == 1) & (y_valid_pred == 0))
        C_est[1, 0] = np.sum((y == 0) & (y_valid_pred == 1))
        C_est[1, 1] = np.sum((y == 1) & (y_valid_pred == 1))
        C_est = C_est / len(y)
        v_est = np.array([labeled_y0, labeled_y1])

        target_predicted_y0 = np.sum(y_unlabeled_pred == 0) / len(y_unlabeled_pred)
        target_predicted_y1 = np.sum(y_unlabeled_pred == 1) / len(y_unlabeled_pred)

        mu_pred_est = np.array([target_predicted_y0, target_predicted_y1])
        try:
            w_est = np.matmul(np.linalg.inv(C_est), mu_pred_est)
        except np.linalg.LinAlgError as ex:
            # confusion matrix not invertible
            # so we bail out without completing bbsc
            # print(C_est)
            return clf, unlabeled_valid_pct_pos
        assert w_est.shape == (2,), w_est.shape
        mu_est = np.matmul(np.diag(v_est), w_est)
        assert mu_est.shape == (2,), mu_est.shape

        w_est_nn = w_est.clip(
            0
        )  # w_est_nn is the non-negative version of w_est, clipping class weights to 0
        class_weights = {0: w_est_nn[0], 1: w_est_nn[1]}

        sigma_min = np.min(np.linalg.eigvals(C_est))
        # print(f"KS-test p={p:.3f}, predicted pos% = {mu_est[1]*100:.2f}% (raw pred pos% = {target_predicted_y1*100:.2f}%), class weights = {class_weights}, σ_min = {sigma_min:.3f}")

        if p > 0.01 or sigma_min <= 0.05:
            # don't use BBSC if no skew detected between labeled validation and unlabeled validation sets
            return clf, unlabeled_valid_pct_pos

        bbsc_clf = sklearn.linear_model.LogisticRegression(
            solver="liblinear", penalty="l2", class_weight=class_weights
        )
        bbsc_clf.fit(X, y)
        return bbsc_clf, target_predicted_y1

    return clf, unlabeled_valid_pct_pos
