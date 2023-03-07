# model source: https://fasttext.cc/docs/en/crawl-vectors.html

import fasttext.util
import pandas as pd
import numpy as np
from gensim.models.fasttext import FastText

"""Downloading of the model:
fasttext.util.download_model("en", if_exists="ignore")  # English
ft = fasttext.load_model("cc.en.300.bin")
ft.save_model("cc.en.300.bin")
"""


class createFastText:
    """class for creation of Fast Text vectorization technique"""

    def fit(self):
        """pass, necessary for HiClass implementation"""
        pass

    def sent_vec(self, sent: list) -> np.ndarray:
        """vectorizes a sentence (creates vector representation for a sentence)

        Args:
            sent list: input sentence

        Returns:
            sentence vector representation list: transformed input sentence, vector representation of it
        """
        vector_size = 300
        wv_res = np.zeros(vector_size)
        ctr = 1
        for w in sent:
            ctr += 1
            wv_res += self.model.get_word_vector(w)
        wv_res = wv_res / ctr
        return wv_res

    def transform(self, x_train: pd.Series, x_test: pd.Series) -> list:
        """transforms the input train and test text into vector representations

        Args:
            x_train pd.Series: input train text
            x_test pd.Series: input test text

        Returns:
            X_train, X_test list: transformed train and test text input
        """
        self.x_train = x_train
        self.x_test = x_test

        # tokenization of the data
        self.x_train = self.x_train.apply(lambda x: [word for word in x.split()])
        self.x_test = self.x_test.apply(lambda x: [word for word in x.split()])

        fasttext.util.download_model(
            "en", if_exists="ignore"
        )  # downloading and saving once
        ft.save_model("cc.en.300.bin")
        ft = fasttext.load_model("cc.en.300.bin")
        self.model = ft

        # X_train
        self.x_train = self.x_train.apply(self.sent_vec)
        X_train = self.x_train.to_list()

        # X_test
        self.x_test = self.x_test.apply(self.sent_vec)
        X_test = self.x_test.to_list()
        return X_train, X_test
