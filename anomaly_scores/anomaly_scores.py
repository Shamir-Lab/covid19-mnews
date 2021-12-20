''' Anomaly scores used as unsupervised features '''
from anomaly_scores.lof import *
from anomaly_scores.ocsvm import *
from anomaly_scores.isolation_forest import *


def add_anomaly_scores_seen(X_train, y_train, numerical_cols, methods_vec, standardization):
    """
    Calculates anomaly scores and adds them as unsupervised features to the training set.

    Args:
        X_train: Training set
        y_train: Training labels
        numerical_cols: A list of numerical feature names to standardize, if standardization==True.
        methods_vec: A binary vector representing anomaly detection approaches to run:
                    0: local outlier factor (LOF) - Trained on the entire X_train
                    1: local outlier factor (LOF) - Trained only on negative labels of X_train
                    2: One Class SVM - Trained on the entire X_train
                    3: One Class SVM - Trained only on negative labels of X_train
                    4: Isolation forest
        standardization: A flag indicating whether to standardize the anomaly scores' columns.

    Returns:
        X_train: The training set containing the new columns.
        standard_params: The standardization parameters, used for the test set standardization.
        clf_dict: The anomaly classifiers, used for the test set anomaly detection.
    """

    num_of_methods = 5
    assert len(methods_vec) == num_of_methods, f"Error! the expected length of methods_vec is: {num_of_methods}."

    train_scores, standard_params, clf_dict = {}, {}, {}

    # Generate majority training set - only "negative" training instances.
    if methods_vec[1] or methods_vec[3]:
        neg_indices = y_train[y_train == 0].index
        majority_train = X_train[X_train.index.isin(neg_indices)]
    else:
        majority_train = None

    # LOF
    if methods_vec[0]:
        train_scores["lof_score_all"], clf_dict["lof_all"], standard_params["lof_all"] = local_outlier_factor(
            X_train.copy(), numerical_cols)

    if methods_vec[1]:
        train_scores["lof_score_majority"], clf_dict["lof_majority"], standard_params[
            "lof_majority"] = local_outlier_factor(X_train.copy(), numerical_cols, majority_train=majority_train)

    # One Class SVM
    if methods_vec[2]:
        train_scores["ocsvm_score_all"], clf_dict["ocsvm_all"], standard_params["ocsvm_all"] = one_class_svm(
            X_train.copy(), numerical_cols)

    if methods_vec[3]:
        train_scores["ocsvm_score_majority"], clf_dict["ocsvm_majority"], standard_params[
            "ocsvm_majority"] = one_class_svm(X_train.copy(), numerical_cols, majority_train=majority_train)

    # Isolation forest anomaly
    if methods_vec[4]:
        train_scores["if_anomaly_score"], clf_dict["if_anomaly"] = calculate_IsolationForest_anomaly(X_train)

    # Add anomaly scores to X_train
    anomaly_cols = train_scores.keys()
    for col_name in anomaly_cols:
        X_train[col_name] = train_scores[col_name]

    # Standardize the anomaly columns
    if standardization:
        X_train, standard_params["anomaly_scores"] = standardize_data(X_train, anomaly_cols)

    return X_train, standard_params, clf_dict


def add_anomaly_scores_unseen(X_test, numerical_cols, methods_vec, standard_params, clf_dict, standardization):
    """
    Calculates anomaly scores and adds them as unsupervised features to the test set.

    Args:
        X_test: Test set
        numerical_cols: A list of numerical feature names to standardize, if standardization==True.
        methods_vec: A binary vector representing anomaly detection approaches to run:
                    0: local outlier factor (LOF) - Trained on the entire X_train
                    1: local outlier factor (LOF) - Trained only on negative labels of X_train
                    2: One Class SVM - Trained on the entire X_train
                    3: One Class SVM - Trained only on negative labels of X_train
                    4: Isolation forest
        standard_params: The standardization parameters of the numerical columns of the training set.
        clf_dict: The anomaly classifiers, trained on the training set.
        standardization: A flag indicating whether to standardize the anomaly scores' columns.

    Returns: The test set with the new columns.
    """

    num_of_methods = 5
    assert len(methods_vec) == num_of_methods, f"Error! the expected length of methods_vec is: {num_of_methods}."

    test_scores = {}

    # LOF
    if methods_vec[0]:
        test_scores["lof_score_all"] = local_outlier_factor_unseen(X_test.copy(), clf_dict["lof_all"],
                                                                   numerical_cols, standard_params["lof_all"])

    if methods_vec[1]:
        test_scores["lof_score_majority"] = local_outlier_factor_unseen(X_test.copy(), clf_dict["lof_majority"],
                                                                        numerical_cols,
                                                                        standard_params["lof_majority"])

    # One Class SVM
    if methods_vec[2]:
        test_scores["ocsvm_score_all"] = one_class_svm_unseen(X_test.copy(), clf_dict["ocsvm_all"], numerical_cols,
                                                              standard_params["ocsvm_all"])

    if methods_vec[3]:
        test_scores["ocsvm_score_majority"] = one_class_svm_unseen(X_test.copy(), clf_dict["ocsvm_majority"],
                                                                   numerical_cols,
                                                                   standard_params["ocsvm_majority"])

    # isolation forrest anomaly
    if methods_vec[4]:
        test_scores["if_anomaly_score"] = predict_unseen_IsolationForest_anomaly(X_test, clf_dict["if_anomaly"])

    # add anomaly scores to X_train
    anomaly_cols = test_scores.keys()
    for col_name in anomaly_cols:
        X_test[col_name] = test_scores[col_name]

    # standardize the anomaly columns
    if standardization:
        X_test = standardize_unseen_data(X_test, anomaly_cols, standard_params["anomaly_scores"])

    return X_test
