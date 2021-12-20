from ml_models.ml_models import MLModel
from sklearn import svm


class SvmModel(MLModel):

    def __init__(self, selection_metric='XGB', n_features=100, standardization=True, anomaly_vector=[0, 0, 0, 0, 0],
                 kernel=None):
        if not standardization:
            print("Note! SVM requires data standardization. Hence standardization is switched to True.")
            standardization = True
        MLModel.__init__(self, selection_metric, n_features, standardization, anomaly_vector)

        # Hyperparameters
        self.kernel = kernel

        # Model
        self.model_name = f'SVM ({kernel})'
        self.clf = svm.SVC(kernel=kernel, probability=True)
        print(f"Hyperparameters: {self.clf.get_params()}")

