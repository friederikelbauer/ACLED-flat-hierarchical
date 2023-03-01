"""
Summary: 
    Preprocesses the short and long dataset and splits it into train and test

Returns:
    Preprocesses the data (trimming, shuffling)
    Creates two datafiles each for short and long with train and test data
    
"""

# Imports
from src.settings import Settings
import pandas as pd
from sklearn.model_selection import train_test_split

# file path settings
settings = Settings(_env_file="paths/.env.eda")


class PreprocessingData:
    def __init__(self, data, stopwords):
        self.data = data

    def trim_data(self):
        """removing entries that have to little and to long notes sections"""
        self.data = self.data[self.data["notes"].str.len() > 20]
        self.data = self.data[self.data["notes"].str.len() < 650]

        return self

    def shuffle_data(self):
        """shuffles data"""
        self.data = self.data.sample(frac=1)
        return self

    def give_data(self):
        """returns data"""
        return self.data


class TrainTestSplit:
    def __init__(self, datasize: str) -> None:
        self.datasize = datasize

    def make_split(self):
        """ "reads the data and splits it into train and test"""
        if self.datasize == "short":
            self.data = pd.read_pickle(settings.short_clean)
        if self.datasize == "long":
            self.data = pd.read_pickle(settings.long_clean)

        self.train, self.test = train_test_split(self.data, test_size=0.2)
        return self

    def add_third_level(self) -> pd.DataFrame:
        """adds a third (theoretical) level as first level to the data"""

        # TRAIN
        first_level = []
        for index, row in self.train.iterrows():
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
        self.train["first_level"] = first_level

        # TEST
        first_level = []
        for index, row in self.test.iterrows():
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
        self.test["first_level"] = first_level

        return self.train, self.test


short = pd.read_csv(settings.short)
short_clean = PreprocessingData(short).trim_data().shuffle_data().give_data()

short_clean.to_pickle("data/processed/short_clean.pkl")
short_train, short_test = TrainTestSplit("short").make_split().add_third_level()
# csv
short_train.to_csv("data/processed/short_train.csv")
short_test.to_csv("data/processed/short_test.csv")
# pickle
short_train.to_pickle("data/processed/short_train.pkl")
short_test.to_pickle("data/processed/short_test.pkl")


long = pd.read_csv(settings.long)
long_clean = PreprocessingData(long).trim_data().shuffle_data().give_data()
long_clean.to_pickle("data/processed/long_clean.pkl")
long_train, long_test = TrainTestSplit("long").make_split().add_third_level()

# csv
long_train.to_csv("data/processed/long_train.csv")
long_test.to_csv("data/processed/long_test.csv")
# pickle
long_train.to_pickle("data/processed/long_train.pkl")
long_test.to_pickle("data/processed/long_test.pkl")
