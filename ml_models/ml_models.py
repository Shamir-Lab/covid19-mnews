from data_preprocessing.multivariate_imputation import *
from anomaly_scores.anomaly_scores import *
from feature_selection.feature_selection import *


class MLModel:
    """
    The MLModel object represents a general ML model, capturing the pre-training steps.
    Child classes may have additional attributes, such as specific hyperparameters.
    Usually used after performing CV.
    """

    def __init__(self, selection_metric='XGB', n_features=100, standardization=False, anomaly_vector=[0, 0, 0, 0, 0]):
        # Pre-processing parameters
        self.selection_metric = selection_metric
        self.n_features = n_features
        self.standardization = standardization
        self.standard_params = []

        # Imputer (learned)
        self.categorical_mode = 0
        self.imputer = 0

        # Anomaly Parameters
        self.anomaly_vector = anomaly_vector
        self.std_params_for_anomaly = []
        self.anomaly_clf = {}
        self.anomaly_new_cols = []

        # Model parameters
        self.clf = None
        self.model_name = ''
        self.selected_features = []


    def fit(self, X_train, y_train):
        """ Train model """
        print(f"Train {self.model_name}.\nTraining set size: {len(X_train)}")

        bool_cols = list(X_train.columns[X_train.dtypes == 'bool'])
        numerical_cols = [col for col in X_train.columns if col not in bool_cols]

        # Data imputation
        # Linear interpolation/ffill can be performed earlier to data partition
        self.categorical_mode = X_train[bool_cols].mode().iloc[0]
        X_train[bool_cols] = X_train[bool_cols].fillna(self.categorical_mode)
        X_train, self.imputer = multivariate_imputation_seen(X_train, numerical_cols)

        # Standardization
        if self.standardization:
            X_train, self.standard_params = standardize_data(X_train, numerical_cols)

        # Anomaly scores
        X_train, self.std_params_for_anomaly, self.anomaly_clf = add_anomaly_scores_seen(
            X_train, y_train, numerical_cols, self.anomaly_vector, standardization=self.standardization)
        self.anomaly_new_cols = list(self.anomaly_clf.keys())

        # Feature selection
        self.selected_features = feature_selection(X_train, y_train, selection_metric=self.selection_metric, K=self.n_features)
        X_train = X_train[self.selected_features]

        # Train XGB
        self.clf.fit(X_train, y_train)


    def predict(self, X_test):
        return self.clf.predict_proba(X_test)[:, 1]


    def evaluation(self, X_test, y_test):
        """ Predict and evaluate model """
        print(f"Evaluate {self.model_name}.\nTest set size: {len(X_test)}")

        bool_cols = list(X_test.columns[X_test.dtypes == 'bool'])
        numerical_cols = [col for col in X_test.columns if col not in bool_cols]

        # Data imputation
        # Linear interpolation/ffill can be performed earlier to data partition
        X_test[bool_cols] = X_test[bool_cols].fillna(self.categorical_mode)
        X_test = multivariate_imputation_unseen(X_test, numerical_cols, self.imputer)

        # Standartization
        if self.standardization:
            X_test = standardize_unseen_data(X_test, numerical_cols, self.standard_params)

        # Anomaly scores
        X_test = add_anomaly_scores_unseen(X_test, numerical_cols, self.anomaly_vector, self.std_params_for_anomaly,
                                           self.anomaly_clf, standardization=self.standardization)

        # Feature selection
        X_test = X_test[self.selected_features]

        # Predict
        predict_proba = self.predict(X_test)
        model_results = {}
        model_results[self.model_name] = predict_proba
        model_results["target"] = y_test
        risk_scores_df = pd.DataFrame.from_dict(model_results)

        return risk_scores_df
