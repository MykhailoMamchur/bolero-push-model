import pickle

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import top_k_accuracy_score


class ModelController:
    def __init__(self, target_column='COMPANY_ID', n_estimators=200):
        self.model = None
        self.target_column = target_column
        self.n_estimators = n_estimators

    def save(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)
        print(f"ModelController saved to {filepath}")

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as file:
            data_manager = pickle.load(file)
        print(f"ModelController loaded from {filepath}")
        return data_manager

    def score_model_top_k_accuracy(self, k, dataset):
        assert self.model is not None
        X, y = self._get_subsets(dataset, target_column=self.target_column)
        score = top_k_accuracy_score(y_true=y, y_score=self.model.predict_proba(X), k=k)
        return score

    def _get_subsets(self, dataset, target_column):
        X = dataset.drop(target_column, axis=1)
        y = dataset.loc[:, target_column]
        return X, y

    def model_train(self, dataset: pd.DataFrame):
        self.model = RandomForestClassifier(n_estimators=self.n_estimators)
        X, y = self._get_subsets(dataset, target_column=self.target_column)
        self.model.fit(X=X, y=y)

    def model_predict(self, data: pd.DataFrame):
        assert self.model is not None
        y_pred = self.model.predict_proba(X=data)
        return y_pred
