# feature extraction - making them numbers
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class createTFIDF:
    """class for creation of TF-IDF vectorization technique"""

    def __init__(self):
        """initializes TF-IDF class"""
        self.vect = TfidfVectorizer()

    def fit(self):
        """necessary for HiClass implementation

        Returns:
            TFIDF initialization
        """
        return self.vect

    def transform(self, x_train: pd.Series, x_test: pd.Series):
        """transforms the input train and test text into vector representation

        Args:
            x_train pd.Series: input train text
            x_test pd.Series: input test text

        Returns:
            scipy.sparse._csr.csr_matrix (no type hinting available): transformed text represenation
        """
        self.x_train = x_train
        self.x_test = x_test
        # transformation for bigger values needed
        self.x_train = self.vect.fit_transform(self.x_train.astype("U").values)
        self.x_test = self.vect.transform(self.x_test.astype("U").values)

        return self.x_train, self.x_test
