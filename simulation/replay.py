import pandas as pd
from config import DATA
from processing.preprocess import validate


def load_demo():
    frame = pd.read_csv(DATA / "scenarios" / "trip_20.csv")
    validate(frame)
    return [part.reset_index(drop=True) for _, part in frame.groupby("window_id", sort=True)]
