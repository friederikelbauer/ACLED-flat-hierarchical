import sklearn
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from hiclass.metrics import f1, recall, precision
from src.utils.conversion_functions import *


class ModelEvaluator:
    def __init__(self, y_test: pd.DataFrame, y_pred: pd.DataFrame, name: str):
        self.y_test = y_test
        self.y_pred = y_pred
        self.name = name
        self.labels = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
        ]

        self.categories = [
            "Armed clash",
            "Government regains territory",
            "Non-state actor overtakes territory",
            "Chemical weapon",
            "Air/drone strike",
            "Suicide bomb",
            "Shelling/artillery/missile attack",
            "Remote explosive/landmine/IED",
            "Grenade",
            "Sexual violence",
            "Attack",
            "Abduction/forced disappearance",
            "Peaceful protest",
            "Protest with intervention",
            "Excessive force against protesters",
            "Violent demonstration",
            "Mob violence",
            "Agreement",
            "Arrests",
            "Change to group/activity",
            "Disrupted weapons use",
            "Headquarters or base established",
            "Looting/property destruction",
            "Non-violent transfer of territory",
            "Other",
        ]

    def save_hierarchy_predictions(self, x_test: pd.DataFrame) -> None:
        # save predictions for further analysis
        y_test_1, y_pred_1 = hierarchical_to_flat(self.y_test, self.y_pred, level=1)
        y_test_2, y_pred_2 = hierarchical_to_flat(self.y_test, self.y_pred, level=2)
        output_data = pd.DataFrame(
            {
                "x_test": x_test,
                "y_test_event": y_test_1,
                "y_pred_event": y_pred_1,
                "y_test_sub_event": y_test_2,
                "y_pred_sub_event": y_pred_2,
            },
        )
        output_data.to_csv(f"src/results/prediction_results/{self.name}_outputdata.csv")

    def save_flat_predictions(self, x_test: pd.DataFrame) -> None:
        y_test_1 = flat_to_hierarchical(self.y_test)
        y_pred_1 = flat_to_hierarchical(self.y_pred)

        output_data = pd.DataFrame(
            {
                "x_test": x_test,
                "y_test_event": y_test_1["event_type"].values,
                "y_pred_event": y_pred_1["event_type"].values,
                "y_test_sub_event": self.y_test,
                "y_pred_sub_event": self.y_pred,
            }
        )
        output_data.to_csv(f"src/results/prediction_results/{self.name}_outputdata.csv")

    def make_confusionmatrix(self):
        """function for both hierarchical and flat confusion matrix"""

        mat = confusion_matrix(self.y_test, self.y_pred)
        # normalizing
        cm_normalized = mat.astype("float") / mat.sum(axis=1)[:, np.newaxis]

        greens = sns.cubehelix_palette(start=2, rot=0, dark=0, light=0.95, as_cmap=True)
        fig, ax = plt.subplots(figsize=(9, 9))  # Sample figsize in inches
        sns.heatmap(
            cm_normalized,
            square=True,
            annot=False,  # annot=True,
            linewidths=0.5,
            ax=ax,
            cmap=greens,
            fmt=".2f",  # how many after comma
            xticklabels=self.labels,  # self.categories,
            yticklabels=self.labels,  # self.categories,
        )
        plt.xlabel("true class", fontsize=22)
        plt.ylabel("predicted class", fontsize=22)
        plt.savefig(f"src/results/prediction_results/{self.name}.jpg")

    ####################
    #### FLAT EVALUATION
    def evaluate_flat(self, level: int):
        # if event type is
        if level == 1:
            self.labels = [1, 2, 3, 4, 5, 6]
            self.categories = [
                "Battles",
                "Explosions/Remore violence",
                "Violence against civilians",
                "Protests",
                "Riots",
                "Strategic developments",
            ]

        # flat evaluation
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision_macro = precision_score(self.y_test, self.y_pred, average="macro")
        recall_macro = recall_score(self.y_test, self.y_pred, average="macro")
        f1_macro = f1_score(self.y_test, self.y_pred, average="macro")
        f1_weighted = f1_score(self.y_test, self.y_pred, average="weighted")
        flat_statistics = {
            "time": datetime.now(),
            "name": self.name,
            "level": level,
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall macro": recall_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }

        flat_report = classification_report(
            self.y_test, self.y_pred, target_names=self.categories, output_dict=True
        )  # # , target_names=names --> if order is known okay otherwise it sorts them wrong

        # conversion
        y_true_hierarchical = flat_to_hierarchical(self.y_test)
        y_pred_hierarchical = flat_to_hierarchical(self.y_pred)

        # hierarchical evaluation
        hierarchical_precision = precision(y_true_hierarchical, y_pred_hierarchical)
        hierarchical_f1 = f1(y_true_hierarchical, y_pred_hierarchical)
        hierarchical_recall = recall(y_true_hierarchical, y_pred_hierarchical)

        hierarchical_statistics = {
            "name": "hierarchical" + self.name,
            "hierarchical precision": hierarchical_precision,
            "hierarchical f1": hierarchical_f1,
            "hierarchical recall": hierarchical_recall,
        }

        stats = flat_statistics, flat_report, hierarchical_statistics

        # https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format
        df = pd.DataFrame(stats)
        df.to_csv(f"src/results/prediction_results/{self.name}_{level}.csv")

    #############################
    ###### HIERACHICAL EVALUATION
    def evaluate_hierarchical(
        self,
        hierarchy_level: int,  # event type or sub-event type
        make_matrix: bool = False,
    ):

        if hierarchy_level == 1:
            annotation_categories = [
                "Battles",
                "Explosions/Remore violence",
                "Violence against civilians",
                "Protests",
                "Riots",
                "Strategic developments",
            ]
        if hierarchy_level == 2:
            annotation_categories = self.categories

        # hierarchical evaluation
        hierarchical_precision = precision(self.y_test, self.y_pred)
        hierarchical_f1 = f1(self.y_test, self.y_pred)
        hierarchical_recall = recall(self.y_test, self.y_pred)

        hierarchical_statistics = {
            "name": "hierarchical" + self.name,
            "hierarchical precision": hierarchical_precision,
            "hierarchical f1": hierarchical_f1,
            "hierarchical recall": hierarchical_recall,
        }

        # conversion
        y_true, y_pred = hierarchical_to_flat(
            self.y_test, self.y_pred, level=hierarchy_level
        )

        # flat evaluation
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average="macro")
        recall_macro = recall_score(y_true, y_pred, average="macro")
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        flat_statistics = {
            "time": datetime.now(),
            "name": self.name,
            "level": hierarchy_level,
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall macro": recall_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }

        flat_report = classification_report(
            y_true, y_pred, target_names=annotation_categories, output_dict=True
        )  # # , target_names=names --> if order is known okay otherwise it sorts them wrong

        stats = flat_statistics, flat_report, hierarchical_statistics

        # https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format
        df = pd.DataFrame(stats)
        df.to_csv(f"src/results/prediction_results/{self.name}_{hierarchy_level}.csv")

        # Flat confusion matrix evaluation
        if make_matrix == True:
            # self.y_test = y_true
            # self.y_pred = y_pred
            self.make_confusionmatrix()
