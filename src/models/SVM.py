# models
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import numpy as np


class createSVM:
    def __init__(self, flat: bool = False) -> None:
        """initiliazes the Support Vector Machine with option for wrapper

        Args:
            flat (boolean, optional): used to include wrapper if flat classifiation method chosen. Defaults to False.
        """
        if flat:
            self.classifier = OneVsRestClassifier(
                LinearSVC(C=1.0, penalty="l2", max_iter=1000, dual=False)
            )  # max iter should be at 1000

        else:
            self.classifier = LinearSVC(C=1.0, penalty="l2", max_iter=1000, dual=False)

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
