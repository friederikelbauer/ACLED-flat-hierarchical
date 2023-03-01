import numpy as np
import gensim.downloader as api

"""
Sources: 
1) https://github.com/PradipNichite/Youtube-Tutorials/blob/main/Yotutube_WordVectors.ipynb
2) https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
3) https://radimrehurek.com/gensim/models/word2vec.html
4) https://analyticsindiamag.com/word2vec-vs-glove-a-comparative-guide-to-word-embedding-techniques/
5) https://github.com/codebasics/deep-learning-keras-tf-tutorial/blob/master/42_word2vec_gensim/42_word2vec_gensim.ipynb
6) Sarkar 2016, p.189f.
"""


class createW2V:
    """class for creation of Word2Vec vectorization technique"""

    def __init__(self):
        """initializes the Word2Vec model"""
        self.model = api.load("word2vec-google-news-300")

    def fit(self):
        """pass, necessary for HiClass implementation"""
        pass

    def vectorize_sentence(self, sentence: list) -> np.ndarray:
        """vectorizes the input sentence to calculate vector representation for each sentence

        Args:
            sentence list: input text sentence

        Returns:
            sentence vector representation list: transformed input sentence, vector representation of it
        """
        # fixed vector size used so all sentences are represented with the same length
        vector_size = self.model.vector_size  # 300
        wv_res = np.zeros(vector_size)
        ctr = 0
        for word in sentence:
            if word in self.model:
                ctr += 1
                wv_res += self.model[word]

        # calculating the vector representation of each sentence
        # if no word of the sentence is in the model
        if ctr != 0:
            wv_res = wv_res / ctr  # averaging the vector representation
        else:
            wv_res = wv_res / 1
        return wv_res

    def transform(self, x_train: list, x_test: list) -> list:
        """transforms the input train and test sentences into vector representations

        Args:
            x_train list: input train text
            x_test list: input test text

        Returns:
            X_train, X_test: transformed train and test text input
        """
        self.x_train = x_train
        self.x_test = x_test
        # X_train
        self.x_train = self.x_train.apply(lambda x: [word for word in x.split()])
        self.x_train = self.x_train.apply(self.vectorize_sentence)
        X_train = self.x_train.to_list()
        # X_test
        self.x_test = self.x_test.apply(lambda x: [word for word in x.split()])
        self.x_test = self.x_test.apply(self.vectorize_sentence)
        X_test = self.x_test.to_list()
        return X_train, X_test
