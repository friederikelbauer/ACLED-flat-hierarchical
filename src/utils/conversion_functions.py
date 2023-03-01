"""
Summary: 
    inlcudes conversion functions for the predicted classes. from flat to hierarchical (back-tracking) and hierarchical to flat (cutting)
"""

import pandas as pd


def add_third_level(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    # TRAIN
    first_level = []
    for index, row in train.iterrows():
        if (
            row["event_type"] == "Battles"
            or row["event_type"] == "Explosions/Remote violence"
            or row["event_type"] == "Violence against civilians"
        ):
            first_level.append("Violent events")
        if row["event_type"] == "Protests" or row["event_type"] == "Riots":
            first_level.append("Demonstrations")
        if row["event_type"] == "Strategic developments":
            first_level.append("Non-violent actions")
    train["first_level"] = first_level

    # TEST
    first_level = []
    for index, row in test.iterrows():
        if (
            row["event_type"] == "Battles"
            or row["event_type"] == "Explosions/Remote violence"
            or row["event_type"] == "Violence against civilians"
        ):
            first_level.append("Violent events")
        if row["event_type"] == "Protests" or row["event_type"] == "Riots":
            first_level.append("Demonstrations")
        if row["event_type"] == "Strategic developments":
            first_level.append("Non-violent actions")
    test["first_level"] = first_level

    return train, test


def flat_to_hierarchical(flat: list) -> pd.DataFrame:
    """converts the flat classification class prediction to a hierarchical one by back-tracking the taxonomy

    Args:
        flat input list of predicted classes

    Returns:
        hierarchical: pandas dataframe of the two event types (level 1: event type, level 2: sub-event type)
    """
    hierarchical = pd.DataFrame()

    event_type = []
    sub_event_type = []
    for i in flat:
        sub_event_type.append(i)

        if (
            i == "Armed clash"
            or i == "Government regains territory"
            or i == "Non-state actor overtakes territory"
        ):
            event_type.append("Battles")

        if (
            i == "Chemical weapon"
            or i == "Air/drone strike"
            or i == "Suicide bomb"
            or i == "Shelling/artillery/missile attack"
            or i == "Remote explosive/landmine/IED"
            or i == "Grenade"
        ):
            event_type.append("Explosions/Remote violence")

        if (
            i == "Sexual violence"
            or i == "Attack"
            or i == "Abduction/forced disappearance"
        ):
            event_type.append("Violence against civilians")

        if (
            i == "Peaceful protest"
            or i == "Protest with intervention"
            or i == "Excessive force against protesters"
        ):
            event_type.append("Protests")

        if i == "Violent demonstration" or i == "Mob violence":
            event_type.append("Riots")

        if (
            i == "Agreement"
            or i == "Arrests"
            or i == "Change to group/activity"
            or i == "Disrupted weapons use"
            or i == "Headquarters or base established"
            or i == "Looting/property destruction"
            or i == "Non-violent transfer of territory"
            or i == "Other"
        ):
            event_type.append("Strategic developments")

    hierarchical["event_type"] = event_type
    hierarchical["sub_event_type"] = sub_event_type

    return hierarchical


def checking_consistency(prediction: list) -> None:
    """function to check the taxonomy consistence for hierarchical predictions

    Args:
        prediction (list): _description_
    """
    # testing causal correctness
    for i in prediction:

        if i[1] == "Battles":
            if (
                i[-1] != "Armed clash"
                and i[-1] != "Government regains territory"
                and i[-1] != "Non-state actor overtakes territory"
            ):
                print("something wrong, check", i)

        if i[1] == "Explosions/Remote violence":
            if (
                i[-1] != "Air/drone strike"
                and i[-1] != "Chemical weapon"
                and i[-1] != "Suicide bomb"
                and i[-1] != "Shelling/artillery/missile attack"
                and i[-1] != "Remote explosive/landmine/IED"
                and i[-1] != "Grenade"
            ):
                print("something wrong, check", i)
        if i[1] == "Violence against civilians":
            if (
                i[-1] != "Sexual violence"
                and i[-1] != "Attack"
                and i[-1] != "Abduction/forced disappearance"
            ):
                print("something wrong, check", i)

        if i[1] == "Protests":
            if (
                i[-1] != "Peaceful protest"
                and i[-1] != "Protest with intervention"
                and i[-1] != "Excessive force against protesters"
            ):
                print("something wrong, check", i)

        if i[1] == "Riots":
            if i[-1] != "Violent demonstration" and i[-1] != "Mob violence":
                print("something wrong, check", i)
        if i[1] == "Strategic developments":
            if (
                i[-1] != "Agreement"
                and i[-1] != "Arrests"
                and i[-1] != "Change to group/activity"
                and i[-1] != "Disrupted weapons use"
                and i[-1] != "Headquarters or base established"
                and i[-1] != "Looting/property destruction"
                and i[-1] != "Non-violent transfer of territory"
                and i[-1] != "Other"
            ):
                print("something wrong, check", i)


def hierarchical_to_flat(
    y_test_hierarchical: pd.DataFrame, y_pred_hierarchical: pd.DataFrame, level: int
) -> list:
    """creates the flat representation of the true hierarchical class and the predicted hierarchical class

    Args:
        y_test_hierarchical (pandas DataFrame): true hierarchical class
        y_pred_hierarchical (pandas DataFrame): predicted hierarchical class
        level (int): level needed (level 1: event type, level 2: sub-event type)

    Returns:
        flat representations of the hierarchical true classes and hierarchical predicted classes depending on level
    """
    y_test_flat = []
    y_pred_flat = []

    # event type level
    if level == 1:
        for index, row in y_test_hierarchical["event_type"].iteritems():
            y_test_flat.append(row)
        y_pred_flat = []
        for i in y_pred_hierarchical:
            y_pred_flat.append(i[0])
    # sub-event type level
    if level == 2:
        for index, row in y_test_hierarchical["sub_event_type"].iteritems():
            y_test_flat.append(row)
        y_pred_flat = []
        for i in y_pred_hierarchical:
            y_pred_flat.append(i[-1])

    return y_test_flat, y_pred_flat
