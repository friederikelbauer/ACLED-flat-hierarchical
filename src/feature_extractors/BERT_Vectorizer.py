"""sources:
- https://towardsdatascience.com/feature-extraction-with-bert-for-text-classification-533dde44dc2f
- https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
- https://www.sbert.net/docs/quickstart.html
- https://huggingface.co/bert-base-uncased
- https://github.com/google-research/bert"""

# Imports
import torch
import pandas as pd
import numpy as np
from sentence_transformers import (
    SentenceTransformer,
    models,
)


class createBERTVectorizer:
    """class for creation of BERT Vectorizer"""

    def __init__(self) -> None:
        """initializes the BERT Vectorizer class and assigns available resources"""
        self.device = [
            torch.device("cpu") if torch.cuda.is_available() else torch.device("cuda")
        ]

    def fit(self):
        """necessary for HiClass implementation but passed"""
        pass

    def transform(self, train: pd.Series, test: pd.Series) -> np.ndarray:
        """creates the BERT Vectors

        Args:
            train pd.Series: train text input
            test pd.Series: test text input

        Returns:
            np.ndarray: two ndarrays of transformed train and test text
        """
        # train
        self.train_sentences = [row for index, row in train.iteritems()]

        # test
        self.test_sentences = [row for index, row in test.iteritems()]

        # bert base uncased model
        self.word_embedding_model = models.Transformer(
            "bert-base-uncased", max_seq_length=256
        )
        pooling_model = models.Pooling(
            self.word_embedding_model.get_word_embedding_dimension()
        )  # size of the vectors
        model = SentenceTransformer(modules=[self.word_embedding_model, pooling_model])

        # encoding of text
        train_embeddings = model.encode(self.train_sentences, show_progress_bar=True)
        test_embeddings = model.encode(self.test_sentences, show_progress_bar=True)

        return train_embeddings, test_embeddings
