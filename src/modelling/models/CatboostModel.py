from src.modelling.models.Model import Model
import catboost as cat
import pandas as pd


class CatboostModel(Model):

    def __init__(self, cat_params: {}):
        self.params = cat_params
        self.model = None

    def train_model(self, train_data: pd.DataFrame, train_label: pd.DataFrame):

        model = cat.CatBoostClassifier(**self.params)
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
        # one hot encode
        data = pd.get_dummies(data)

        return data

    @staticmethod
    def _get_categorical_columns(df: pd.DataFrame):
        dtypes = list(df.dtypes)
        cat_features = [i for i in range(len(dtypes)) if str(dtypes[i]) not in {"float64", "int64", "bool"}]
        return cat_features
