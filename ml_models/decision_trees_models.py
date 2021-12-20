from ml_models.ml_models import MLModel
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from catboost import CatBoostClassifier
import xgboost as xgb


class CatboostModel(MLModel):

    def __init__(self, selection_metric='XGB', n_features=100, standardization=False, anomaly_vector=[0, 0, 0, 0, 0],
                 n_estimators=None, depth=None, learning_rate=None, l2_leaf_reg=None):
        MLModel.__init__(self, selection_metric, n_features, standardization, anomaly_vector)

        # Hyperparameters
        self.n_estimators = n_estimators
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg

        # Model
        self.model_name = 'CTB'
        self.clf = CatBoostClassifier(verbose=False,
                                      n_estimators=self.n_estimators,
                                      depth=self.depth,
                                      learning_rate=self.learning_rate,
                                      l2_leaf_reg=self.l2_leaf_reg)
        print(f"Hyperparameters: {self.clf.get_params()}")


class XgboostModel(MLModel):

    def __init__(self, selection_metric='XGB', n_features=100, standardization=False, anomaly_vector=[0, 0, 0, 0, 0],
                 n_estimators=None, max_depth=None, learning_rate=None, colsample_bytree=None):
        MLModel.__init__(self, selection_metric, n_features, standardization, anomaly_vector)

        # Hyperparameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.colsample_bytree = colsample_bytree

        # Model
        self.model_name = 'XGB'
        self.clf = xgb.XGBClassifier(n_estimators=n_estimators,
                                     learning_rate=learning_rate,
                                     max_depth=max_depth,
                                     colsample_bytree=colsample_bytree)
        print(f"Hyperparameters: {self.clf.get_params()}")


class GbtModel(MLModel):

    def __init__(self, selection_metric='XGB', n_features=100, standardization=False, anomaly_vector=[0, 0, 0, 0, 0],
                 n_estimators=None, learning_rate=None, max_depth=None):
        MLModel.__init__(self, selection_metric, n_features, standardization, anomaly_vector)

        # Hyperparameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        # Model
        self.model_name = 'GBT'
        self.clf = GradientBoostingClassifier(n_estimators=n_estimators,
                                              learning_rate=learning_rate,
                                              max_depth=max_depth)
        print(f"Hyperparameters: {self.clf.get_params()}")


class RFModel(MLModel):

    def __init__(self, selection_metric='XGB', n_features=100, standardization=False, anomaly_vector=[0, 0, 0, 0, 0],
                 n_estimators=None, max_depth=None):
        MLModel.__init__(self, selection_metric, n_features, standardization, anomaly_vector)

        # Hyperparameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        # Model
        self.model_name = 'RF'
        self.clf = RandomForestClassifier(n_estimators=n_estimators,
                                          max_depth=max_depth)
        print(f"Hyperparameters: {self.clf.get_params()}")
