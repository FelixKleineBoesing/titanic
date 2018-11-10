import abc
import pandas as pd

class Model(abc.ABC):

    @abc.abstractmethod
    def train_model(self, train_data: pd.DataFrame, train_label: pd.DataFrame):
        pass

    @abc.abstractmethod
    def predict(self, test_data: pd.DataFrame):
        pass

    @staticmethod
    def preprocess(train_data: pd.DataFrame):
        pass
