from src.modelling.models.Model import Model
import lightgbm as lgbm
import pandas as pd


class LightgbmModel(Model):

    def __init__(self, lgbm_params: {}, n_rounds: int):
        self.params = lgbm_params
        self.params["objective"] = "regression"
        self.model = None
        self.n_rounds = n_rounds

    def train_model(self, train_data: pd.DataFrame, train_label: pd.DataFrame):
        train_data = self._preprocess(train_data)
        train_label = train_label.Survived.tolist()
        dtrain = lgbm.Dataset(train_data, label=train_label)

        self.model = lgbm.train(self.params, dtrain, num_boost_round=self.n_rounds)

    def predict(self, data: pd.DataFrame):
        data = self._preprocess(data)
        dtest = lgbm.Dataset(data)
        y_pred = self.model.predict(dtest)
        return y_pred

    @staticmethod
    def _preprocess(data: pd.DataFrame):
        # pass data
        return data

    @staticmethod
    def _get_categorical_columns(df: pd.DataFrame):
        dtypes = list(df.dtypes)
        cat_features = [i for i in range(len(dtypes)) if str(dtypes[i]) not in {"float64", "int64", "bool"}]
        return cat_features