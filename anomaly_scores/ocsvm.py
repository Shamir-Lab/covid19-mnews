''' Anomaly scores used as unsupervised features '''
from data_preprocessing.standardization import standardize_data, standardize_unseen_data, is_data_standardized
from sklearn.svm import OneClassSVM


def one_class_svm(X_train, standardized_features, majority_train=None):
    """
    Trains a one class SVM on the training set to generate anomaly features.
    Performs standardization (required).

    Args:
        X_train: Training set.
        standardized_features:A list of feature names to standardized.
        majority_train: A DF that, if given, X_train is replaced with. Contains negative examples only.

    Returns:
        ocsvm_score: OCSVM scores.
        ocsvm_clf: OCSVM classifier.
        features_params: The standardization parameters for X_train (mean, SD).
    """
    # Check if data is already standardized
    standard_params = []
    if not is_data_standardized(X_train, standardized_features):
        print("Perform standardization for one class svm")
        X_train, standard_params = standardize_data(X_train, standardized_features)

    # Fit classifier on majority_train or on X_train
    ocsvm_clf = OneClassSVM()
    if majority_train is None:
        ocsvm_clf.fit(X_train)
    else:
        ocsvm_clf.fit(majority_train)

    ocsvm_score = ocsvm_clf.score_samples(X_train)

    return ocsvm_score, ocsvm_clf, standard_params


def one_class_svm_unseen(X_test, ocsvm_clf, standardized_features, features_params):
    """
    Applies one class SVM on the test set.

    Args:
        X_test: Test set
        ocsvm_clf: Trained OCSVM classifier.
        standardized_features:  A list of numerical features to standardize.
        features_params: The standardization parameters of X_train

    Returns: The OCSVM scores for the test set.
    """
    # Check if data is already standardized
    if features_params:
        X_test = standardize_unseen_data(X_test, standardized_features, features_params)

    ocsvm_score = ocsvm_clf.score_samples(X_test)

    return ocsvm_score
