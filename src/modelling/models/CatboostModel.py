from src.modelling.models.Model import Model
import catboost as cat
import pandas as pd


class CatboostModel(Model):

    def __init__(self, cat_params: {}, n_rounds: int):
        self.params = cat_params
        self.model = None
        self.n_rounds = n_rounds

    def train_model(self, train_data: pd.DataFrame, train_label: pd.DataFrame):

        self.model = cat.CatBoostClassifier(**self.params, num_boost_round=self.n_rounds)
        train_data = self._preprocess(train_data)
        train_label = train_label.Survived.tolist()
        pool = cat.Pool(train_data, train_label)

        self.model.fit(pool, verbose=False)


    def predict(self, data: pd.DataFrame):
        data = self._preprocess(data)
        dtest = cat.Pool(data)
        y_pred = self.model.predict(dtest)
        return y_pred

    @staticmethod
    def _preprocess(data: pd.DataFrame):
        # pass dat
        return data

    @staticmethod
    def _get_categorical_columns(df: pd.DataFrame):
        dtypes = list(df.dtypes)
        cat_features = [i for i in range(len(dtypes)) if str(dtypes[i]) not in {"float64", "int64", "bool"}]
        return cat_features
