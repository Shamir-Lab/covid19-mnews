from ml_models.ml_models import MLModel
from sklearn.naive_bayes import GaussianNB


class NBModel(MLModel):

    def __init__(self, selection_metric='XGB', n_features=100, anomaly_vector=[0, 0, 0, 0, 0]):
        MLModel.__init__(self, selection_metric, n_features, anomaly_vector)

        # Model
        self.model_name = 'NB'
        self.clf = GaussianNB()
