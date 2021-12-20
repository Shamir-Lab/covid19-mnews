''' Data imputation - multivariate Iterative Imputation, inspired by MICE '''
import sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def multivariate_imputation_seen(X_train, columns):
    """
    Perform multivariate iterative imputation for SEEN data (training set).
    Should be done for train and test separately to avoid data leakage.

    Args:
        X_train: Training set.
        columns: Numerical columns to be imputed

    Returns:
        X_train: Imputed training set.
        imputer: Fitted imputation model for test set imputation
    """
    imputer = IterativeImputer()
    X_train[columns] = imputer.fit_transform(X_train[columns])

    return X_train, imputer


def multivariate_imputation_unseen(X_test, columns, imputer):
    """
    Perform multivariate iterative imputation for UNSEEN data (test set).
    Should be done for train and test separately to avoid data leakage.

    Args:
        X_test: Test set.
        columns: Numerical columns to be imputed
        imputer: Fitted imputation model.

    Returns: X_test: Imputed training set.
    """
    assert type(imputer) == sklearn.impute.IterativeImputer, "Error! Invalid imputer's type"
    X_test[columns] = imputer.transform(X_test[columns])

    return X_test
