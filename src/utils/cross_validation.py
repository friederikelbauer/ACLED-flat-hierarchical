"""
Summary: 
    performs cross validation for either flat or hierarchical text classification

Creates:
    for each cross validation round a file with the predicted labels for manual analysis
"""

# Imports
from sklearn.model_selection import KFold
from datetime import datetime
import pandas as pd
import fasttext.util

# hierarchical wrapper
from hiclass import (
    LocalClassifierPerParentNode,
)

# metrics
from hiclass.metrics import f1, recall, precision
from src.utils.conversion_functions import *
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def flat_cross_validate(
    local_classifier,
    vectorizer,
    X: pd.DataFrame,
    y: pd.DataFrame,
    name: str,
) -> None:

    """performs cross validation for the flat text classification models

    Args:
        local_classifier (): classifier used for the prediction
        vectorizer (): vectorizer used for the prediction
        X (pandas DataFrame): input x data
        y (pandas DataFrame): input y data
        name (string): name used for the produced csv file, should include level, classifier and vectorizer info

    Creates:
        csv file with the flatt cross validation results
    """

    # flat evaluation
    accuracy_scores = []
    precision_macros = []
    recall_macros = []
    f1_macro_scores = []
    f1_weighted_scores = []

    # hierarchical evaluation
    hierarchical_precision_scores = []
    hierarchical_f1_scores = []
    hierarchical_recall_scores = []

    # times
    times = []

    # how many folds used
    kf = KFold(n_splits=5)

    round_number = 0
    # iterating over each fold
    for train_index, test_index in kf.split(X):
        round_number += 1
        start = datetime.now()

        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # vectorize
        X_train, X_test = vectorizer.transform(x_train, x_test)
        # classify
        y_pred = local_classifier.fit_classifier(X_train, y_train, X_test)
        end = datetime.now()

        # writing y pred to file for manual examination
        output_data = pd.DataFrame(
            {"x_test": x_test, "y_test": y_test, "y_pred": y_pred}
        )
        output_data.to_csv(
            f"src/results/cross_validation/{name}_outputdata_{round_number}.csv"
        )

        # flat evaluation
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_macros.append(precision_score(y_test, y_pred, average="macro"))
        recall_macros.append(recall_score(y_test, y_pred, average="macro"))
        f1_macro_scores.append(f1_score(y_test, y_pred, average="macro"))
        f1_weighted_scores.append(f1_score(y_test, y_pred, average="weighted"))

        # conversion to hierarchical
        y_true_hierarchical = flat_to_hierarchical(y_test)
        y_pred_hierarchical = flat_to_hierarchical(y_pred)

        hierarchical_precision_scores.append(
            precision(y_true_hierarchical, y_pred_hierarchical)
        )
        hierarchical_f1_scores.append(f1(y_true_hierarchical, y_pred_hierarchical))
        hierarchical_recall_scores.append(
            recall(y_true_hierarchical, y_pred_hierarchical)
        )
        total_time = end - start
        times.append(total_time.total_seconds())

    results = {
        "accuracy": accuracy_scores,
        "precision_macro": precision_macros,
        "recall macro": recall_macros,
        "f1 macro": f1_macro_scores,
        "f1 weighted": f1_weighted_scores,
        "hierarchical recall": hierarchical_recall_scores,
        "hierarchical f1": hierarchical_f1_scores,
        "hierarchical precision": hierarchical_precision_scores,
        "times": times,
    }
    df = pd.DataFrame(results)
    df.to_csv(f"src/results/cross_validation/{name}.csv")


def hierarchical_cross_validate(
    local_classifier, vectorizer, X: pd.DataFrame, y: pd.DataFrame, name: str
) -> None:
    """performs cross validation for the hierarchical text classification models

    Args:
        local_classifier (): classifier used for the prediction
        vectorizer (): vectorizer used for the prediction
        X (pandas DataFrame): input x data
        y (pandas DataFrame): input y data
        name (string): name used for the produced csv file, should include level, classifier and vectorizer info

    Creates:
        csv file with the flatt cross validation results
    """

    hierarchy_classifier = LocalClassifierPerParentNode(
        local_classifier=local_classifier
    )

    # flat evaluation - LEVEL 1
    accuracy_scores_level1 = []
    precision_macros_level1 = []
    recall_macros_level1 = []
    f1_macro_scores_level1 = []
    f1_weighted_scores_level1 = []

    # flat evaluation - LEVEL 2
    accuracy_scores_level2 = []
    precision_macros_level2 = []
    recall_macros_level2 = []
    f1_macro_scores_level2 = []
    f1_weighted_scores_level2 = []

    # hierarchical evaluation
    hierarchical_precision_scores = []
    hierarchical_f1_scores = []
    hierarchical_recall_scores = []

    # times
    times = []

    # how many folds used
    kf = KFold(n_splits=5)

    round_number = 0
    # iterating over each fold
    for train_index, test_index in kf.split(X):
        round_number += 1
        start = datetime.now()

        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # vectorize
        X_train, X_test = vectorizer.transform(x_train, x_test)
        # classify
        hierarchy_classifier.fit(X_train, y_train)
        y_pred = hierarchy_classifier.predict(X_test)
        end = datetime.now()
        y_pred = [list(map(str, lst)) for lst in y_pred]

        hierarchical_precision_scores.append(precision(y_test, y_pred))
        hierarchical_f1_scores.append(f1(y_test, y_pred))
        hierarchical_recall_scores.append(recall(y_test, y_pred))
        ###############
        # conversion to flat - LEVEL 1
        y_test_flat, y_pred_flat = hierarchical_to_flat(y_test, y_pred, level=1)

        # flat evaluation
        accuracy_scores_level1.append(accuracy_score(y_test_flat, y_pred_flat))
        precision_macros_level1.append(
            precision_score(y_test_flat, y_pred_flat, average="macro")
        )

        recall_macros_level1.append(
            recall_score(y_test_flat, y_pred_flat, average="macro")
        )

        f1_macro_scores_level1.append(
            f1_score(y_test_flat, y_pred_flat, average="macro")
        )
        f1_weighted_scores_level1.append(
            f1_score(y_test_flat, y_pred_flat, average="weighted")
        )
        total_time = end - start
        times.append(total_time.total_seconds())

        ################
        # conversion to flat - LEVEL 2
        y_test_flat, y_pred_flat = hierarchical_to_flat(y_test, y_pred, level=2)

        # flat evaluation
        accuracy_scores_level2.append(accuracy_score(y_test_flat, y_pred_flat))
        precision_macros_level2.append(
            precision_score(y_test_flat, y_pred_flat, average="macro")
        )

        recall_macros_level2.append(
            recall_score(y_test_flat, y_pred_flat, average="macro")
        )

        f1_macro_scores_level2.append(
            f1_score(y_test_flat, y_pred_flat, average="macro")
        )
        f1_weighted_scores_level2.append(
            f1_score(y_test_flat, y_pred_flat, average="weighted")
        )
        # writing y pred to file for manual examination
        output_data = pd.DataFrame(
            {"x_test": x_test, "y_test": y_test_flat, "y_pred": y_pred_flat}
        )
        output_data.to_csv(
            f"src/results/cross_validation/{name}_outputdata_{round_number}.csv"
        )

    results_level1 = {
        "accuracy": accuracy_scores_level1,
        "precision_macro": precision_macros_level1,
        "recall macro": recall_macros_level1,
        "f1 macro": f1_macro_scores_level1,
        "f1 weighted": f1_weighted_scores_level1,
        "hierarchical recall": hierarchical_recall_scores,
        "hierarchical f1": hierarchical_f1_scores,
        "hierarchical precision": hierarchical_precision_scores,
        "times": times,
    }
    df_level1 = pd.DataFrame(results_level1)
    df_level1.to_csv(f"src/results/cross_validation/{name}_level1.csv")

    results_level2 = {
        "accuracy": accuracy_scores_level2,
        "precision_macro": precision_macros_level2,
        "recall macro": recall_macros_level2,
        "f1 macro": f1_macro_scores_level2,
        "f1 weighted": f1_weighted_scores_level2,
        "hierarchical recall": hierarchical_recall_scores,
        "hierarchical f1": hierarchical_f1_scores,
        "hierarchical precision": hierarchical_precision_scores,
        "times": times,
    }
    df_level2 = pd.DataFrame(results_level2)
    df_level2.to_csv(f"src/results/cross_validation/{name}_level2.csv")
