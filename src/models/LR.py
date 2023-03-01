from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


class createLogisticRegression:
    def __init__(self) -> None:
        """initializes the Logistic Regression"""
        self.classifier = LogisticRegression(
            penalty="l2",  # default
            C=1.0,  # default
            solver="lbfgs",  # default
            multi_class="multinomial",
            max_iter=10000,
        )

    def get_classifier(self):
        """used for hierarchical classifier, just returns classifier"""
        return self.classifier

    def fit_classifier(
        self,
        x_train,
        y_train: pd.Series,
        x_test,
    ) -> np.ndarray:
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
