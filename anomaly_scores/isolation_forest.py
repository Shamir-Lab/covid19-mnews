''' Isolation Forest - anomaly detection '''
from sklearn.ensemble import IsolationForest

def calculate_IsolationForest_anomaly(X_train):
    """
    Trains an Isolation Forest model on the training set, and generates a score for each case.

    Args:
        X_train: Training set.

    Returns:
        anomaly_score: The IsolationForest scores.
        clf: The IsolationForest classifier.
    """
    clf = IsolationForest()
    clf.fit(X_train)
    anomaly_score = clf.decision_function(X_train)

    return anomaly_score, clf


def predict_unseen_IsolationForest_anomaly(X_test, clf):
    """
    Applies Isolation Forest to the test set.

    Args:
        X_test: Test set.
        clf: The Isolation Forest classifier that was trained on the training set.

    Returns:
        anomaly_score: The Isolation Forest scores of each case in the test set.
    """
    anomaly_score = clf.decision_function(X_test)

    return anomaly_score
