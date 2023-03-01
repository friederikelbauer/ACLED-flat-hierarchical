# models
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


class createRF:
    def __init__(self) -> None:
        """initiliazes the Random Forest"""
        self.classifier = RandomForestClassifier()

    def get_classifier(self):
        """used for hierarchical classifier, just returns classifier"""
        return self.classifier

    def fit_classifier(self, x_train, y_train: pd.Series, x_test) -> np.ndarray:
        """
        fits the classifier to input and predicts y_pred

        Args:
            x_train scipy.sparse._csr.csr_matrix: input x_train
            y_train pd.Series: input y_train
            x_test scipy.sparse._csr.csr_matrix: input x_test

        Returns:
            y_pred np.ndarray: prediction
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.classifier.fit(self.x_train, self.y_train)
        y_pred = self.classifier.predict(self.x_test)
        return y_pred
