"""
Summary: 
    Script for creating datasets

Returns:
    Takes the raw downloaded data from the ACLED data page and 
    creates two sizes of datasets (short and long)
"""

# Imports
import pandas as pd
from src.settings import Settings
import os
from pathlib import Path

settings = Settings(_env_file="paths/.env.eda")

# reading raw data
data_1997_2009 = pd.read_csv(settings.DATA_1997_2009)
data_2010 = pd.read_csv(settings.DATA_2010)
data_2011 = pd.read_csv(settings.DATA_2011)
data_2012 = pd.read_csv(settings.DATA_2012)
data_2013 = pd.read_csv(settings.DATA_2013)
data_2014 = pd.read_csv(settings.DATA_2014)
data_2015 = pd.read_csv(settings.DATA_2015)
data_2016 = pd.read_csv(settings.DATA_2016)
data_2017 = pd.read_csv(settings.DATA_2017)
data_2018 = pd.read_csv(settings.DATA_2018)
data_2019 = pd.read_csv(settings.DATA_2019)
data_2020 = pd.read_csv(settings.DATA_2020)
data_2021 = pd.read_csv(settings.DATA_2021)
data_2022 = pd.read_csv(settings.DATA_2022)

data = pd.concat(
    [
        data_1997_2009,
        data_2010,
        data_2011,
        data_2012,
        data_2013,
        data_2014,
        data_2015,
        data_2016,
        data_2017,
        data_2018,
        data_2019,
        data_2020,
        data_2021,
        data_2022,
    ]
)

# creating different lengths of datasets
data = data[["notes", "event_type", "sub_event_type"]]
data = data.dropna().drop_duplicates()
data.to_csv("data/processed/1997-01-01-2022-12-31.csv")  # full dataset

# creating different sizes of datasets dataset
def sample_dataset(data: pd.DataFrame, length: int):
    try:
        peaceful_protest = data[data["sub_event_type"] == "Peaceful protest"].sample(
            length
        )
    except ValueError:
        peaceful_protest = data[data["sub_event_type"] == "Peaceful protest"]
    try:
        armed_clash = data[data["sub_event_type"] == "Armed clash"].sample(length)
    except ValueError:
        armed_clash = data[data["sub_event_type"] == "Armed clash"]
    try:
        attack = data[data["sub_event_type"] == "Attack"].sample(length)
    except ValueError:
        attack = data[data["sub_event_type"] == "Attack"]
    try:
        shelling = data[
            data["sub_event_type"] == "Shelling/artillery/missile attack"
        ].sample(length)
    except ValueError:
        shelling = data[data["sub_event_type"] == "Shelling/artillery/missile attack"]
    try:
        drone = data[data["sub_event_type"] == "Air/drone strike"].sample(length)
    except ValueError:
        drone = data[data["sub_event_type"] == "Air/drone strike"]
    try:
        violent_protest = data[
            data["sub_event_type"] == "Violent demonstration"
        ].sample(length)
    except ValueError:
        violent_protest = data[data["sub_event_type"] == "Violent demonstration"]
    try:
        mob = data[data["sub_event_type"] == "Mob violence"].sample(length)
    except ValueError:
        mob = data[data["sub_event_type"] == "Mob violence"]
    try:
        landmine = data[
            data["sub_event_type"] == "Remote explosive/landmine/IED"
        ].sample(length)
    except ValueError:

        landmine = data[data["sub_event_type"] == "Remote explosive/landmine/IED"]
    try:
        protest_intervention = data[
            data["sub_event_type"] == "Protest with intervention"
        ].sample(length)
    except ValueError:
        protest_intervention = data[
            data["sub_event_type"] == "Protest with intervention"
        ]
    try:
        looting = data[data["sub_event_type"] == "Looting/property destruction"].sample(
            length
        )
    except ValueError:
        looting = data[data["sub_event_type"] == "Looting/property destruction"]
    try:
        abduction = data[
            data["sub_event_type"] == "Abduction/forced disappearance"
        ].sample(length)
    except ValueError:
        abduction = data[data["sub_event_type"] == "Abduction/forced disappearance"]
    try:
        group = data[data["sub_event_type"] == "Change to group/activity"].sample(
            length
        )
    except ValueError:
        group = data[data["sub_event_type"] == "Change to group/activity"]
    try:
        arrests = data[data["sub_event_type"] == "Arrests"].sample(length)
    except ValueError:
        arrests = data[data["sub_event_type"] == "Arrests"]
    try:
        government = data[
            data["sub_event_type"] == "Government regains territory"
        ].sample(length)
    except ValueError:
        government = data[data["sub_event_type"] == "Government regains territory"]
    try:
        other = data[data["sub_event_type"] == "Other"].sample(length)
    except ValueError:
        other = data[data["sub_event_type"] == "Other"]
    try:
        disrupted = data[data["sub_event_type"] == "Disrupted weapons use"].sample(
            length
        )
    except ValueError:
        disrupted = data[data["sub_event_type"] == "Disrupted weapons use"]
    try:
        non_state = data[
            data["sub_event_type"] == "Non-state actor overtakes territory"
        ].sample(length)
    except ValueError:
        non_state = data[
            data["sub_event_type"] == "Non-state actor overtakes territory"
        ]
    try:
        grenade = data[data["sub_event_type"] == "Grenade"].sample(length)
    except ValueError:
        grenade = data[data["sub_event_type"] == "Grenade"]
    try:
        force = data[
            data["sub_event_type"] == "Excessive force against protesters"
        ].sample(length)
    except ValueError:
        force = data[data["sub_event_type"] == "Excessive force against protesters"]
    try:
        transfer = data[
            data["sub_event_type"] == "Non-violent transfer of territory"
        ].sample(length)
    except ValueError:
        transfer = data[data["sub_event_type"] == "Non-violent transfer of territory"]
    try:
        sexual_violence = data[data["sub_event_type"] == "Sexual violence"].sample(
            length
        )
    except ValueError:
        sexual_violence = data[data["sub_event_type"] == "Sexual violence"]
    try:
        agreement = data[data["sub_event_type"] == "Agreement"].sample(length)
    except ValueError:
        agreement = data[data["sub_event_type"] == "Agreement"]
    try:
        suicide = data[data["sub_event_type"] == "Suicide bomb"].sample(length)
    except ValueError:
        suicide = data[data["sub_event_type"] == "Suicide bomb"]
    try:
        headquarters = data[
            data["sub_event_type"] == "Headquarters or base established"
        ].sample(length)
    except ValueError:
        headquarters = data[
            data["sub_event_type"] == "Headquarters or base established"
        ]
    chemical = data[data["sub_event_type"] == "Chemical weapon"]

    # concatenating
    data = pd.concat(
        [
            peaceful_protest,
            armed_clash,
            attack,
            shelling,
            drone,
            violent_protest,
            mob,
            landmine,
            protest_intervention,
            looting,
            abduction,
            group,
            arrests,
            government,
            other,
            disrupted,
            non_state,
            grenade,
            force,
            transfer,
            sexual_violence,
            agreement,
            suicide,
            headquarters,
            chemical,
        ]
    )

    return data


def check_path(path: str) -> None:
    """
    creates path if it does not exist
    """
    if not os.path.exists(path):
        Path(path).mkdir(parents=True, exist_ok=True)


"""
There are thwo sizes of datasets:
short = sample of the data for faster running of the models
long = all of the data used in the analysis, for long running of the models
"""

check_path("data/processed")

# short data
data_short = sample_dataset(data=data, length=10000)
data_short.to_csv("data/processed/short.csv")  # ~183k

data_long = sample_dataset(data=data, length=57990)

data_long.to_csv("data/processed/long.csv")  # same size as Piskorski et al. size

# create all paths for predictions
check_path("src/results/prediction_results/lr")
check_path("src/results/prediction_results/svm")
check_path("src/results/prediction_results/rf")

# create all paths for cross-validation
check_path("src/results/cross_validation/lr")
check_path("src/results/cross_validation/svm")
check_path("src/results/cross_validation/rf")
