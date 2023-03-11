import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots

plt.style.use("science")


def create_predictions_plot(classifier: str, mode: str) -> None:
    """creates a prediction plot

    Args:
        classifier str: type of classifier
        mode str: either flat or hierarchical
    """

    if mode == "flat":
        end_path = "_level2_outputdata.csv"
    else:
        end_path = "_outputdata.csv"

    # getting the data
    # count vectorizer
    count_data = pd.read_csv(
        f"src/results/prediction_results/{classifier}/{classifier}_{mode}_count_vectorizer{end_path}"
    )
    count_predicted = count_data["y_pred_sub_event"].value_counts().to_dict()
    actual = count_data["y_test_sub_event"].value_counts().to_dict()
    actual_classes = [
        actual["Armed clash"],
        actual["Government regains territory"],
        actual["Non-state actor overtakes territory"],
        actual["Chemical weapon"],
        actual["Air/drone strike"],
        actual["Suicide bomb"],
        actual["Shelling/artillery/missile attack"],
        actual["Remote explosive/landmine/IED"],
        actual["Grenade"],
        actual["Sexual violence"],
        actual["Attack"],
        actual["Abduction/forced disappearance"],
        actual["Peaceful protest"],
        actual["Protest with intervention"],
        actual["Excessive force against protesters"],
        actual["Violent demonstration"],
        actual["Mob violence"],
        actual["Agreement"],
        actual["Arrests"],
        actual["Change to group/activity"],
        actual["Disrupted weapons use"],
        actual["Headquarters or base established"],
        actual["Looting/property destruction"],
        actual["Non-violent transfer of territory"],
        actual["Other"],
    ]

    countvectorizer = [
        count_predicted["Armed clash"],
        count_predicted["Government regains territory"],
        count_predicted["Non-state actor overtakes territory"],
        count_predicted["Chemical weapon"],
        count_predicted["Air/drone strike"],
        count_predicted["Suicide bomb"],
        count_predicted["Shelling/artillery/missile attack"],
        count_predicted["Remote explosive/landmine/IED"],
        count_predicted["Grenade"],
        count_predicted["Sexual violence"],
        count_predicted["Attack"],
        count_predicted["Abduction/forced disappearance"],
        count_predicted["Peaceful protest"],
        count_predicted["Protest with intervention"],
        count_predicted["Excessive force against protesters"],
        count_predicted["Violent demonstration"],
        count_predicted["Mob violence"],
        count_predicted["Agreement"],
        count_predicted["Arrests"],
        count_predicted["Change to group/activity"],
        count_predicted["Disrupted weapons use"],
        count_predicted["Headquarters or base established"],
        count_predicted["Looting/property destruction"],
        count_predicted["Non-violent transfer of territory"],
        count_predicted["Other"],
    ]
    # TF-IDF
    tfidf_data = pd.read_csv(
        f"src/results/prediction_results/{classifier}/{classifier}_{mode}_tfidf{end_path}"
    )
    tfidf_predicted = (
        tfidf_data["y_pred_sub_event"].value_counts(normalize=True).to_dict()
    )
    tfidf = [
        tfidf_predicted["Armed clash"],
        tfidf_predicted["Government regains territory"],
        tfidf_predicted["Non-state actor overtakes territory"],
        tfidf_predicted["Chemical weapon"],
        tfidf_predicted["Air/drone strike"],
        tfidf_predicted["Suicide bomb"],
        tfidf_predicted["Shelling/artillery/missile attack"],
        tfidf_predicted["Remote explosive/landmine/IED"],
        tfidf_predicted["Grenade"],
        tfidf_predicted["Sexual violence"],
        tfidf_predicted["Attack"],
        tfidf_predicted["Abduction/forced disappearance"],
        tfidf_predicted["Peaceful protest"],
        tfidf_predicted["Protest with intervention"],
        tfidf_predicted["Excessive force against protesters"],
        tfidf_predicted["Violent demonstration"],
        tfidf_predicted["Mob violence"],
        tfidf_predicted["Agreement"],
        tfidf_predicted["Arrests"],
        tfidf_predicted["Change to group/activity"],
        tfidf_predicted["Disrupted weapons use"],
        tfidf_predicted["Headquarters or base established"],
        tfidf_predicted["Looting/property destruction"],
        tfidf_predicted["Non-violent transfer of territory"],
        tfidf_predicted["Other"],
    ]
    # word2vec
    word2vec_data = pd.read_csv(
        f"src/results/prediction_results/{classifier}/{classifier}_{mode}_word2vec{end_path}"
    )
    word2vec_predicted = (
        word2vec_data["y_pred_sub_event"].value_counts(normalize=True).to_dict()
    )
    word2vec = [
        word2vec_predicted["Armed clash"],
        word2vec_predicted["Government regains territory"],
        word2vec_predicted["Non-state actor overtakes territory"],
        word2vec_predicted["Chemical weapon"],
        word2vec_predicted["Air/drone strike"],
        word2vec_predicted["Suicide bomb"],
        word2vec_predicted["Shelling/artillery/missile attack"],
        word2vec_predicted["Remote explosive/landmine/IED"],
        word2vec_predicted["Grenade"],
        word2vec_predicted["Sexual violence"],
        word2vec_predicted["Attack"],
        word2vec_predicted["Abduction/forced disappearance"],
        word2vec_predicted["Peaceful protest"],
        word2vec_predicted["Protest with intervention"],
        word2vec_predicted["Excessive force against protesters"],
        word2vec_predicted["Violent demonstration"],
        word2vec_predicted["Mob violence"],
        word2vec_predicted["Agreement"],
        word2vec_predicted["Arrests"],
        word2vec_predicted["Change to group/activity"],
        word2vec_predicted["Disrupted weapons use"],
        word2vec_predicted["Headquarters or base established"],
        word2vec_predicted["Looting/property destruction"],
        word2vec_predicted["Non-violent transfer of territory"],
        word2vec_predicted["Other"],
    ]
    # Fast Text
    fasttext_data = pd.read_csv(
        f"src/results/prediction_results/{classifier}/{classifier}_{mode}_fasttext{end_path}"
    )
    fasttext_predicted = (
        fasttext_data["y_pred_sub_event"].value_counts(normalize=True).to_dict()
    )
    fasttext = [
        fasttext_predicted["Armed clash"],
        fasttext_predicted["Government regains territory"],
        fasttext_predicted["Non-state actor overtakes territory"],
        fasttext_predicted["Chemical weapon"],
        fasttext_predicted["Air/drone strike"],
        fasttext_predicted["Suicide bomb"],
        fasttext_predicted["Shelling/artillery/missile attack"],
        fasttext_predicted["Remote explosive/landmine/IED"],
        fasttext_predicted["Grenade"],
        fasttext_predicted["Sexual violence"],
        fasttext_predicted["Attack"],
        fasttext_predicted["Abduction/forced disappearance"],
        fasttext_predicted["Peaceful protest"],
        fasttext_predicted["Protest with intervention"],
        fasttext_predicted["Excessive force against protesters"],
        fasttext_predicted["Violent demonstration"],
        fasttext_predicted["Mob violence"],
        fasttext_predicted["Agreement"],
        fasttext_predicted["Arrests"],
        fasttext_predicted["Change to group/activity"],
        fasttext_predicted["Disrupted weapons use"],
        fasttext_predicted["Headquarters or base established"],
        fasttext_predicted["Looting/property destruction"],
        fasttext_predicted["Non-violent transfer of territory"],
        fasttext_predicted["Other"],
    ]
    # BERT Vectorizer
    bert_data = pd.read_csv(
        f"src/results/prediction_results/{classifier}/{classifier}_{mode}_bert_vectorizer{end_path}"
    )
    bert_predicted = (
        bert_data["y_pred_sub_event"].value_counts(normalize=True).to_dict()
    )
    bert = [
        bert_predicted["Armed clash"],
        bert_predicted["Government regains territory"],
        bert_predicted["Non-state actor overtakes territory"],
        bert_predicted["Chemical weapon"],
        bert_predicted["Air/drone strike"],
        bert_predicted["Suicide bomb"],
        bert_predicted["Shelling/artillery/missile attack"],
        bert_predicted["Remote explosive/landmine/IED"],
        bert_predicted["Grenade"],
        bert_predicted["Sexual violence"],
        bert_predicted["Attack"],
        bert_predicted["Abduction/forced disappearance"],
        bert_predicted["Peaceful protest"],
        bert_predicted["Protest with intervention"],
        bert_predicted["Excessive force against protesters"],
        bert_predicted["Violent demonstration"],
        bert_predicted["Mob violence"],
        bert_predicted["Agreement"],
        bert_predicted["Arrests"],
        bert_predicted["Change to group/activity"],
        bert_predicted["Disrupted weapons use"],
        bert_predicted["Headquarters or base established"],
        bert_predicted["Looting/property destruction"],
        bert_predicted["Non-violent transfer of territory"],
        bert_predicted["Other"],
    ]

    feature_extractors = {
        "Count Vetorizer": countvectorizer,
        "TF-IDF": tfidf,
        "Word2Vec": word2vec,
        "Fast Text": fasttext,
        "BERT Vectorizer": bert,
    }

    x = np.arange(1, 26)  # the label locations
    width = 0.18  # the width of the bars
    multiplier = -1

    fig, ax = plt.subplots(constrained_layout=True)
    ax.bar(
        x + 0.18,
        actual_classes,
        color="white",
        alpha=1,
        width=width * 5.5,
        edgecolor="black",
        linewidth=1.4,
    )

    for attribute, measurement in feature_extractors.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)

        multiplier += 1

    ax.set_ylabel("frequency of prediction in percent", fontsize=18)
    ax.set_xlabel("sub-event class", fontsize=18)
    ax.set_xticks(x + width, np.arange(1, 26), fontsize=18)
    ax.legend(loc="upper left", ncols=5, fontsize=18)

    plt.show()


def create_correct_predictions_plot(classifier: str, mode: str) -> None:
    """creates a prediction plot of the correctly classified predictions

    Args:
        classifier str: type of classifier
        mode str: either flat or hierarchical
    """

    if mode == "flat":
        end_path = "_level2_outputdata.csv"
    else:
        end_path = "_outputdata.csv"

    # getting the data
    # count vectorizer
    count_data = pd.read_csv(
        f"src/results/prediction_results/{classifier}/{classifier}_{mode}_count_vectorizer{end_path}"
    )
    actual = count_data["y_test_sub_event"].value_counts(normalize=True).to_dict()

    count_data["correct"] = np.where(
        count_data["y_pred_sub_event"] == count_data["y_test_sub_event"],
        True,
        None,
    )
    count_data.dropna(inplace=True)
    count_predicted = (
        count_data["y_pred_sub_event"].value_counts(normalize=True).to_dict()
    )

    # actuall class

    actual_classes = [
        actual["Armed clash"],
        actual["Government regains territory"],
        actual["Non-state actor overtakes territory"],
        actual["Chemical weapon"],
        actual["Air/drone strike"],
        actual["Suicide bomb"],
        actual["Shelling/artillery/missile attack"],
        actual["Remote explosive/landmine/IED"],
        actual["Grenade"],
        actual["Sexual violence"],
        actual["Attack"],
        actual["Abduction/forced disappearance"],
        actual["Peaceful protest"],
        actual["Protest with intervention"],
        actual["Excessive force against protesters"],
        actual["Violent demonstration"],
        actual["Mob violence"],
        actual["Agreement"],
        actual["Arrests"],
        actual["Change to group/activity"],
        actual["Disrupted weapons use"],
        actual["Headquarters or base established"],
        actual["Looting/property destruction"],
        actual["Non-violent transfer of territory"],
        actual["Other"],
    ]

    countvectorizer = [
        count_predicted["Armed clash"],
        count_predicted["Government regains territory"],
        count_predicted["Non-state actor overtakes territory"],
        count_predicted["Chemical weapon"],
        count_predicted["Air/drone strike"],
        count_predicted["Suicide bomb"],
        count_predicted["Shelling/artillery/missile attack"],
        count_predicted["Remote explosive/landmine/IED"],
        count_predicted["Grenade"],
        count_predicted["Sexual violence"],
        count_predicted["Attack"],
        count_predicted["Abduction/forced disappearance"],
        count_predicted["Peaceful protest"],
        count_predicted["Protest with intervention"],
        count_predicted["Excessive force against protesters"],
        count_predicted["Violent demonstration"],
        count_predicted["Mob violence"],
        count_predicted["Agreement"],
        count_predicted["Arrests"],
        count_predicted["Change to group/activity"],
        count_predicted["Disrupted weapons use"],
        count_predicted["Headquarters or base established"],
        count_predicted["Looting/property destruction"],
        count_predicted["Non-violent transfer of territory"],
        count_predicted["Other"],
    ]
    # TF-IDF
    tfidf_data = pd.read_csv(
        f"src/results/prediction_results/{classifier}/{classifier}_{mode}_tfidf{end_path}"
    )

    tfidf_data["correct"] = np.where(
        tfidf_data["y_pred_sub_event"] == tfidf_data["y_test_sub_event"],
        True,
        None,
    )

    tfidf_data.dropna(inplace=True)

    tfidf_predicted = (
        tfidf_data["y_pred_sub_event"].value_counts(normalize=True).to_dict()
    )

    tfidf = [
        tfidf_predicted["Armed clash"],
        tfidf_predicted["Government regains territory"],
        tfidf_predicted["Non-state actor overtakes territory"],
        tfidf_predicted["Chemical weapon"],
        tfidf_predicted["Air/drone strike"],
        tfidf_predicted["Suicide bomb"],
        tfidf_predicted["Shelling/artillery/missile attack"],
        tfidf_predicted["Remote explosive/landmine/IED"],
        tfidf_predicted["Grenade"],
        tfidf_predicted["Sexual violence"],
        tfidf_predicted["Attack"],
        tfidf_predicted["Abduction/forced disappearance"],
        tfidf_predicted["Peaceful protest"],
        tfidf_predicted["Protest with intervention"],
        tfidf_predicted["Excessive force against protesters"],
        tfidf_predicted["Violent demonstration"],
        tfidf_predicted["Mob violence"],
        tfidf_predicted["Agreement"],
        tfidf_predicted["Arrests"],
        tfidf_predicted["Change to group/activity"],
        tfidf_predicted["Disrupted weapons use"],
        tfidf_predicted["Headquarters or base established"],
        tfidf_predicted["Looting/property destruction"],
        tfidf_predicted["Non-violent transfer of territory"],
        tfidf_predicted["Other"],
    ]
    # word2vec
    word2vec_data = pd.read_csv(
        f"src/results/prediction_results/{classifier}/{classifier}_{mode}_word2vec{end_path}"
    )
    word2vec_data["correct"] = np.where(
        word2vec_data["y_pred_sub_event"] == word2vec_data["y_test_sub_event"],
        True,
        None,
    )

    word2vec_data.dropna(inplace=True)

    word2vec_predicted = (
        word2vec_data["y_pred_sub_event"].value_counts(normalize=True).to_dict()
    )
    word2vec = [
        word2vec_predicted["Armed clash"],
        word2vec_predicted["Government regains territory"],
        word2vec_predicted["Non-state actor overtakes territory"],
        word2vec_predicted["Chemical weapon"],
        word2vec_predicted["Air/drone strike"],
        word2vec_predicted["Suicide bomb"],
        word2vec_predicted["Shelling/artillery/missile attack"],
        word2vec_predicted["Remote explosive/landmine/IED"],
        word2vec_predicted["Grenade"],
        word2vec_predicted["Sexual violence"],
        word2vec_predicted["Attack"],
        word2vec_predicted["Abduction/forced disappearance"],
        word2vec_predicted["Peaceful protest"],
        word2vec_predicted["Protest with intervention"],
        word2vec_predicted["Excessive force against protesters"],
        word2vec_predicted["Violent demonstration"],
        word2vec_predicted["Mob violence"],
        word2vec_predicted["Agreement"],
        word2vec_predicted["Arrests"],
        word2vec_predicted["Change to group/activity"],
        word2vec_predicted["Disrupted weapons use"],
        word2vec_predicted["Headquarters or base established"],
        word2vec_predicted["Looting/property destruction"],
        word2vec_predicted["Non-violent transfer of territory"],
        word2vec_predicted["Other"],
    ]
    # Fast Text
    fasttext_data = pd.read_csv(
        f"src/results/prediction_results/{classifier}/{classifier}_{mode}_fasttext{end_path}"
    )
    fasttext_data["correct"] = np.where(
        fasttext_data["y_pred_sub_event"] == fasttext_data["y_test_sub_event"],
        True,
        None,
    )

    fasttext_data.dropna(inplace=True)

    fasttext_predicted = (
        fasttext_data["y_pred_sub_event"].value_counts(normalize=True).to_dict()
    )
    fasttext = [
        fasttext_predicted["Armed clash"],
        fasttext_predicted["Government regains territory"],
        fasttext_predicted["Non-state actor overtakes territory"],
        fasttext_predicted["Chemical weapon"],
        fasttext_predicted["Air/drone strike"],
        fasttext_predicted["Suicide bomb"],
        fasttext_predicted["Shelling/artillery/missile attack"],
        fasttext_predicted["Remote explosive/landmine/IED"],
        fasttext_predicted["Grenade"],
        fasttext_predicted["Sexual violence"],
        fasttext_predicted["Attack"],
        fasttext_predicted["Abduction/forced disappearance"],
        fasttext_predicted["Peaceful protest"],
        fasttext_predicted["Protest with intervention"],
        fasttext_predicted["Excessive force against protesters"],
        fasttext_predicted["Violent demonstration"],
        fasttext_predicted["Mob violence"],
        fasttext_predicted["Agreement"],
        fasttext_predicted["Arrests"],
        fasttext_predicted["Change to group/activity"],
        fasttext_predicted["Disrupted weapons use"],
        fasttext_predicted["Headquarters or base established"],
        fasttext_predicted["Looting/property destruction"],
        fasttext_predicted["Non-violent transfer of territory"],
        fasttext_predicted["Other"],
    ]
    # BERT Vectorizer
    bert_data = pd.read_csv(
        f"src/results/prediction_results/{classifier}/{classifier}_{mode}_bert_vectorizer{end_path}"
    )
    bert_data["correct"] = np.where(
        bert_data["y_pred_sub_event"] == bert_data["y_test_sub_event"],
        True,
        None,
    )

    bert_data.dropna(inplace=True)

    bert_predicted = (
        bert_data["y_pred_sub_event"].value_counts(normalize=True).to_dict()
    )
    bert = [
        bert_predicted["Armed clash"],
        bert_predicted["Government regains territory"],
        bert_predicted["Non-state actor overtakes territory"],
        bert_predicted["Chemical weapon"],
        bert_predicted["Air/drone strike"],
        bert_predicted["Suicide bomb"],
        bert_predicted["Shelling/artillery/missile attack"],
        bert_predicted["Remote explosive/landmine/IED"],
        bert_predicted["Grenade"],
        bert_predicted["Sexual violence"],
        bert_predicted["Attack"],
        bert_predicted["Abduction/forced disappearance"],
        bert_predicted["Peaceful protest"],
        bert_predicted["Protest with intervention"],
        bert_predicted["Excessive force against protesters"],
        bert_predicted["Violent demonstration"],
        bert_predicted["Mob violence"],
        bert_predicted["Agreement"],
        bert_predicted["Arrests"],
        bert_predicted["Change to group/activity"],
        bert_predicted["Disrupted weapons use"],
        bert_predicted["Headquarters or base established"],
        bert_predicted["Looting/property destruction"],
        bert_predicted["Non-violent transfer of territory"],
        bert_predicted["Other"],
    ]

    feature_extractors = {
        "Count Vetorizer": countvectorizer,
        "TF-IDF": tfidf,
        "Word2Vec": word2vec,
        "Fast Text": fasttext,
        "BERT Vectorizer": bert,
    }

    x = np.arange(1, 26)  # the label locations
    width = 0.18  # the width of the bars
    multiplier = -1

    fig, ax = plt.subplots(constrained_layout=True)
    ax.bar(
        x + 0.18,
        actual_classes,
        color="white",
        alpha=1,
        width=width * 5.5,
        edgecolor="black",
        linewidth=1.4,
    )

    for attribute, measurement in feature_extractors.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)

        multiplier += 1

    ax.set_ylabel("frequency of correct prediction in percent", fontsize=18)
    ax.set_xlabel("sub-event class", fontsize=18)
    ax.set_xticks(x + width, np.arange(1, 26), fontsize=18)
    ax.legend(loc="upper left", ncols=5, fontsize=18)

    plt.show()


create_correct_predictions_plot(classifier="lr", mode="flat")
