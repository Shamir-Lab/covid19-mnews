from ml_models.ml_models import MLModel
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, SGDClassifier


class LogRegModel(MLModel):

    def __init__(self, selection_metric='XGB', n_features=100, standardization=False, anomaly_vector=[0, 0, 0, 0, 0],
                 penalty=None):
        MLModel.__init__(self, selection_metric, n_features, standardization, anomaly_vector)

        # Hyperparameters
        self.penalty = penalty

        # Model
        self.model_name = f'{penalty} LogReg'
        self.clf = LogisticRegression(penalty=penalty, solver="saga")
        print(f"Hyperparameters: {self.clf.get_params()}")


class LassoModel(MLModel):

    def __init__(self, selection_metric='XGB', n_features=100, standardization=True, anomaly_vector=[0, 0, 0, 0, 0]):
        if not standardization:
            print("Note! Lasso requires data standardization. Hence standardization is switched to True.")
            standardization = True
        MLModel.__init__(self, selection_metric, n_features, standardization, anomaly_vector)

        # Model
        self.model_name = 'Lasso'
        self.clf = Lasso()

    def predict(self, X_test):
        return self.clf.predict(X_test)


class RidgeModel(MLModel):

    def __init__(self, selection_metric='XGB', n_features=100, standardization=True, anomaly_vector=[0, 0, 0, 0, 0]):
        if not standardization:
            print("Note! Lasso requires data standardization. Hence standardization is switched to True.")
            standardization = True
        MLModel.__init__(self, selection_metric, n_features, standardization, anomaly_vector)

        # Model
        self.model_name = 'Ridge'
        self.clf = Ridge()

    def predict(self, X_test):
        return self.clf.predict(X_test)
