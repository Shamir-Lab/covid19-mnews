''' feature selection module '''
import pandas as pd
import xgboost as xgb


def feature_selection(X_train, y_train, selection_metric='', K=100):
    """
    Feature selection according to a given metric.

    Args:
        X_train: Training set
        y_train: Training labels
        selection_metric: Selection metric: 'Correlation' / 'XGB' / pre-defined list (literature review).
        K: Number of features to select

    Returns: A list containing K selected features.
    """
    if type(selection_metric) == list:
        return selection_metric

    if selection_metric == 'Correlation':
        return feature_selection_corr(X_train, y_train, K)

    if selection_metric == 'XGB':
        return feature_selection_xgb(X_train, y_train, K)

    # No selection
    return X_train.columns


def feature_selection_corr(X_train, y_train, K):
    """
    Returns a list of K features with the highest correlation to labels
    """
    corr = X_train.apply(lambda x: x.corr(y_train))
    features = list(corr.nlargest(K).index)

    return features


def feature_selection_xgb(X_train, y_train, K):
    """
    Returns a list of K features with highest importance score according to XGBoost.
    """
    xgb_clf = xgb.XGBClassifier(n_estimators=100)
    xgb_clf.fit(X_train, y_train)
    feature_importance = pd.Series(xgb_clf.feature_importances_, index=X_train.columns)
    features = list(feature_importance.nlargest(K).index)

    return features
