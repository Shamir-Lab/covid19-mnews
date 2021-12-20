''' Local outlier factor - anomaly detection '''
from data_preprocessing.standardization import standardize_data, standardize_unseen_data, is_data_standardized
from sklearn.neighbors import LocalOutlierFactor


def local_outlier_factor(X_train, standardized_features, majority_train=None):
    """
    Trains a Local Outlier Factor on the training set to generate anomaly features.
    Performs standardization (required).

    Args:
        X_train: Training set.
        standardized_features: A list of feature names to standardized.
        majority_train: A DF that, if given, X_train is replaced with. Contains negative examples only.

    Returns:
        lof_score: lof scores.
        lof_clf: lof classifier, to be passed to local_outlier_factor_unseen.
        standard_params: The standardization parameters of X_train (mean, SD).
    """
    # Check if data is already standardized
    standard_params = []
    if not is_data_standardized(X_train, standardized_features):
        print("Performs standardization for LOF")
        X_train, standard_params = standardize_data(X_train, standardized_features)

    # Fit model to majority_train or to X_train
    lof_clf = LocalOutlierFactor(novelty=True)
    if majority_train is None:
        lof_clf.fit(X_train)
    else:
        lof_clf.fit(majority_train)

    lof_score = lof_clf.score_samples(X_train)

    return lof_score, lof_clf, standard_params


def local_outlier_factor_unseen(X_test, lof_clf, standardized_features, standard_params):
    """
    Applies trained Local Outlier Factor on the test set.

    Args:
        X_test: Test set
        lof_clf: Trained lof classifier.
        standardized_features: A list of numerical features to standardize.
        standard_params: The standardization parameters of X_train.

    Returns: The lof scores for the test set.
    """
    # Check if data is already standardized
    if standard_params:
        X_test = standardize_unseen_data(X_test, standardized_features, standard_params)

    lof_score = lof_clf.score_samples(X_test)
    return lof_score
