from src.etl.ETL import ETL
from src.modelling.models.XGBoostModel import XGBoostModel
from src.modelling.models.LightgbmModel import LightgbmModel
from src.modelling.models.CatboostModel import CatboostModel
from src.modelling.Evaluator import Evaluator
from src.modelling.CalculateThreshold import calculate_class

import pandas as pd

# TODO implement catboost & lightgbm models
# TODO implement outlier cleaning
# TODO think about furrther feature preparation


class TitanicChallenge:
    def __init__(self):
        self.measures = None
        self.models = {}
        self.predictions = {}
        self.evaluation = {}

    def apply_etl(self):
        etl = ETL()
        etl.extract_data("../initial_data/train.csv",
                         "../initial_data/test.csv")
        etl.clean_data()
        etl.load_data("../computed_data/train_data.csv",
                      "../computed_data/train_label.csv",
                      "../computed_data/test_data.csv",
                      "../computed_data/validation_label.csv",
                      "../computed_data/validation_data.csv")

    def train_models(self):
        # Load data
        train_data = pd.read_csv("../computed_data/train_data.csv")
        train_label = pd.read_csv("../computed_data/train_label.csv")

        # train with different models
        xgb_model = XGBoostModel(xgb_params={"eta": 0.1, "objective": "binary:logistic"}, n_rounds=20)
        xgb_model.train_model(train_data, train_label)
        self.models["xgb"] = xgb_model

        cat_model = CatboostModel(cat_params={"eta": 0.1}, n_rounds=20)
        cat_model.train_model(train_data, train_label)
        self.models["cat"] = cat_model

        lgb_model = LightgbmModel(lgbm_params={"eta": 0.1}, n_rounds=20)
        lgb_model.train_model(train_data, train_label)
        self.models["lgb"] = lgb_model

    def predict_models(self):
        validation_data = pd.read_csv("../computed_data/validation_data.csv")
        y_pred = self.models["xgb"].predict(validation_data)
        self.predictions["xgb"] = calculate_class(y_pred, "naive")

    def evaluate_models(self):
        assert len(self.predictions) > 0, "call train and predict first!"
        validation_label = pd.read_csv("../computed_data/validation_label.csv")
        for key in self.predictions.keys():
            eval = Evaluator(self.predictions[key], validation_label)
            eval.evaluate_predictions()
            self.evaluation[key] = eval.evaluation

if __name__=="__main__":
    titanic = TitanicChallenge()
    titanic.apply_etl()

    titanic.train_models()
    titanic.predict_models()
    titanic.evaluate_models()
    print(titanic.evaluation["xgb"]["Accuracy"])
