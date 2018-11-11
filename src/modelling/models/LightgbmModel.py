from src.modelling.models.Model import Model
import lightgbm as lgbm
import pandas as pd


class LightgbmModel(Model):

    def __init__(self, xgb_params: {}, n_rounds: int):
        self.params = xgb_params
        self.model = None
        self.n_rounds = n_rounds

    def train_model(self, train_data: pd.DataFrame, train_label: pd.DataFrame):
        train_data = self._preprocess(train_data)
        train_label = train_label.Survived.tolist()
        print(1)
        dtrain = xgb.DMatrix(train_data, train_label)

        self.model = xgb.train(self.params, dtrain, num_boost_round=self.n_rounds)

    def predict(self, data: pd.DataFrame):
        data = self._preprocess(data)
        dtest = xgb.DMatrix(data)
        y_pred = self.model.predict(dtest)
        return y_pred

    @staticmethod
    def _preprocess(data: pd.DataFrame):
        # one hot encode
        data = pd.get_dummies(data)

        return data

