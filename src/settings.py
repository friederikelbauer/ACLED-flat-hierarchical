from pydantic import BaseSettings


class Settings(BaseSettings):
    DATA_1997_2009: str
    DATA_2010: str
    DATA_2011: str
    DATA_2012: str
    DATA_2013: str
    DATA_2014: str
    DATA_2015: str
    DATA_2016: str
    DATA_2017: str
    DATA_2018: str
    DATA_2019: str
    DATA_2020: str
    DATA_2021: str
    DATA_2022: str
    short: str
    short_clean: str
    long: str
    long_clean: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
