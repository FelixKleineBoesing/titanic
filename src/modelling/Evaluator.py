import pandas as pd
import abc
import numpy as np
from sklearn.metrics import confusion_matrix


class Evaluator:

    def __init__(self, pred: pd.DataFrame, actuals: pd.DataFrame):

        test = actuals.Survived.as_matrix()
        self.pred = pred
        self.actuals = actuals.Survived.as_matrix()
        self.con_matrix = None
        self.precision = None
        self.recall = None
        self.f_one_score = None
        self.accuracy = None
        self.evaluation = {}

    def evaluate_predictions(self):
        self._get_confusion_matrix()
        self._get_accuracy()
        self._get_precision()
        self._get_recall()
        self._get_f_one_score()
        self.return_eval_dict()

    def _get_confusion_matrix(self):
        self.con_matrix = confusion_matrix(self.actuals, self.pred)

    def _get_accuracy(self):
        self.accuracy = sum(self.pred == self.actuals) / len(self.pred)

    def _get_recall(self):
        self.recall = sum(np.logical_and(self.pred == 1, self.actuals == 1)) / (sum(np.logical_and(self.pred == 1, self.actuals == 1)) +
            sum(np.logical_and(self.pred == 1, self.actuals == 0)))

    def _get_precision(self):
        tp = self.pred == 1
        self.precision = sum(np.logical_and(self.pred == 1, self.actuals == 1)) / (sum(np.logical_and(self.pred == 1, self.actuals == 1)) +
            sum(np.logical_and(self.pred == 0, self.actuals == 1)))

    def _get_f_one_score(self):
        if self.recall is None:
            self._get_recall()
        if self.precision is None:
            self._get_precision()

        self.f_one_score = (2 * self.precision * self.recall) / (self.precision + self.recall)

    def return_eval_dict(self):

        self.evaluation["F1"] = self.f_one_score
        self.evaluation["Recall"] = self.recall
        self.evaluation["Precision"] = self.precision
        self.evaluation["Confusion Matrix"] = self.con_matrix
        self.evaluation["Accuracy"] = self.accuracy
