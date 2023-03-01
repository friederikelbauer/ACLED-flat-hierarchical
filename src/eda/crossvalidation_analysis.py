"""
script used for crossvalidation analysis
"""

# Imports
import numpy as np
import pandas as pd


def analyse_crossvalidation(datapath: str) -> pd.DataFrame:
    """analyses the crossvalidation results

    Args:
        datapath str: path to data

    Returns:
        pd.DataFrame: returns an analysis of the input data, mean standard deviation and variance of data columns
    """
    data = pd.read_csv(datapath)
    data = data.drop(data.columns[0], axis=1)  # not including the index column

    means = []
    stds = []
    vars = []
    for column in data:
        means.append(np.mean(data[column]))
        stds.append(np.std(data[column]))
        vars.append(np.var(data[column]))

    results = {
        "metric": [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1 macro",
            "f1 weighted",
            "hierarchical recall",
            "hierarchical f1",
            "hierarchical precision",
            "times",
        ],
        "mean": means,
        "standard deviation": stds,
        "variance": vars,
    }

    results_df = pd.DataFrame.from_dict(
        results,
    )
    return results_df


######
# LR
######

# FLAT
flat_lr_count = analyse_crossvalidation(
    "src/results/cross_validation/lr/flat_cross_validation_lr_countvectorizer_level2.csv"
)

flat_lr_tfidf = analyse_crossvalidation(
    "src/results/cross_validation/lr/flat_cross_validation_lr_tfidf_level2.csv"
)
flat_lr_word2vec = analyse_crossvalidation(
    "src/results/cross_validation/lr/flat_cross_validation_lr_word2vec_level2.csv"
)

flat_lr_fasttext = analyse_crossvalidation(
    "src/results/cross_validation/lr/flat_cross_validation_lr_fasttext_level2.csv"
)
flat_lr_bert = analyse_crossvalidation(
    "src/results/cross_validation/lr/flat_cross_validation_lr_bert_vectorizer_level2.csv"
)

## HIERARCHICAL

hier_lr_count = analyse_crossvalidation(
    "src/results/cross_validation/lr/hierarchical_cross_validation_lr_count_vectorizer_level2.csv"
)
hier_lr_tfidf = analyse_crossvalidation(
    "src/results/cross_validation/lr/hierarchical_cross_validation_lr_tfidf_level2.csv"
)
hier_lr_word2vec = analyse_crossvalidation(
    "src/results/cross_validation/lr/hierarchical_cross_validation_lr_word2vec_level2.csv"
)
hier_lr_fasttext = analyse_crossvalidation(
    "src/results/cross_validation/lr/hierarchical_cross_validation_lr_fasttext_level2.csv"
)
hier_lr_bert = analyse_crossvalidation(
    "src/results/cross_validation/lr/hierarchical_cross_validation_lr_bert_vectorizer_level2.csv"
)

######
# RF
######

# FLAT
flat_rf_count = analyse_crossvalidation(
    "src/results/cross_validation/rf/flat_cross_validation_rf_count_vectorizer_level2.csv"
)

flat_rf_tfidf = analyse_crossvalidation(
    "src/results/cross_validation/rf/flat_cross_validation_rf_tfidf_level2.csv"
)

flat_rf_word2vec = analyse_crossvalidation(
    "src/results/cross_validation/rf/flat_cross_validation_rf_word2vec_level2.csv"
)
flat_rf_fasttext = analyse_crossvalidation(
    "src/results/cross_validation/rf/flat_cross_validation_rf_fasttext_level2.csv"
)

flat_rf_bert = analyse_crossvalidation(
    "src/results/cross_validation/rf/flat_cross_validation_rf_bert_vectorizer_level2.csv"
)

# HIERARCHICAL
hier_rf_count = analyse_crossvalidation(
    "src/results/cross_validation/rf/hierarchical_crossvalidation_rf_countvectorizer_level2.csv"
)
hier_rf_tfidf = analyse_crossvalidation(
    "src/results/cross_validation/rf/hierarchical_crossvalidation_rf_tfidf_level2.csv"
)

hier_rf_fasttext = analyse_crossvalidation(
    "src/results/cross_validation/rf/hierarchical_cross_validation_rf_fasttext_level2.csv"
)
hier_rf_word2vec = analyse_crossvalidation(
    "src/results/cross_validation/rf/hierarchical_cross_validation_rf_word2vec_level2.csv"
)
hier_rf_bert = analyse_crossvalidation(
    "src/results/cross_validation/rf/hierarchical_cross_validation_rf_bert_vectorizer_level2.csv"
)


######
# SVM
######

# FLAT
flat_svm_count = analyse_crossvalidation(
    "src/results/cross_validation/svm/flat_cross_validation_svm_count_vectorizer_level2.csv"
)
flat_svm_tfidf = analyse_crossvalidation(
    "src/results/cross_validation/svm/flat_cross_validation_svm_tfidf_level2.csv"
)
flat_svm_word2vec = analyse_crossvalidation(
    "src/results/cross_validation/svm/flat_cross_validation_svm_word2vec_level2.csv"
)
flat_svm_fasttext = analyse_crossvalidation(
    "src/results/cross_validation/svm/flat_cross_validation_svm_fasttext_level2.csv"
)
flat_svm_bert = analyse_crossvalidation(
    "src/results/cross_validation/svm/flat_cross_validation_svm_bert_vectorizer_level2.csv"
)

# HIERARCHICAL
svm_hier_count = analyse_crossvalidation(
    "src/results/cross_validation/svm/hierarchical_cross_validation_svm_count_vectorizer_level2.csv"
)
svm_hier_tfidf = analyse_crossvalidation(
    "src/results/cross_validation/svm/hierarchical_cross_validation_svm_tfidf_level2.csv"
)
svm_hier_word2vec = analyse_crossvalidation(
    "src/results/cross_validation/svm/hierarchical_cross_validation_svm_word2vec_level2.csv"
)
svm_hier_fasttext = analyse_crossvalidation(
    "src/results/cross_validation/svm/hierarchical_cross_validation_svm_fasttext_level2.csv"
)
svm_hier_bert = analyse_crossvalidation(
    "src/results/cross_validation/svm/hierarchical_cross_validation_svm_bert_vectorizer_level2.csv"
)
