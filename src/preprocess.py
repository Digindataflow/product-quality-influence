import pandas as pd
from joblib import dump
from sklearn.ensemble import IsolationForest

from errors import ToMuchNAValueError

CEMENT_TYPES = ["Type I-II", "Type III"]
CEMENT_TYPE_COLUMN = "material_name"
NULL_PERCENTAGE = 10.0
RANSOM_SEED = 0

class Preprocessor:
    def __init__(self, df: pd.DataFrame, steps=None) -> None:
        self.df = df
        self.steps = steps
        self.cleaned_df = None

    def _set_index(self):
        """Set sample date as index"""
        self.df["sample_date"] = pd.to_datetime(self.df.sample_date)
        self.cleaned_df = self.df.set_index("sample_date", drop=True)

    def _check_missing_value(self):
        """check missing value percentage"""
        null_pcg = self.cleaned_df.isnull().mean() * 100
        if (null_pcg > NULL_PERCENTAGE).any():
            columns = null_pcg[null_pcg > NULL_PERCENTAGE]
            raise ToMuchNAValueError(f"To much Na values in {columns}")

    def run(self):
        self._set_index()

        self._check_missing_value()

        if self.steps:
            for step in self.steps:
                self.cleaned_df = step(self.cleaned_df)

    def save(self, path):
        dump(self.cleaned_df, path)

class InterpolateMissingValue:
    """
    interpolate missing value with adjacent values, according
    to datetime index
    """
    def _split_data_by_cement_type(self, df):
        df_type12 = df.loc[df[CEMENT_TYPE_COLUMN] == CEMENT_TYPES[0], :].drop(CEMENT_TYPE_COLUMN, axis=1)
        df_type3 = df.loc[df[CEMENT_TYPE_COLUMN] == CEMENT_TYPES[1], :].drop(CEMENT_TYPE_COLUMN, axis=1)
        return df_type12, df_type3

    def _interpolate_missing_value(self, df):
        df = df.interpolate("index")
        return df.dropna(axis=0 ,how="any")

    def __call__(self, df):
        df_type12, df_type3 = self._split_data_by_cement_type(df)
        df_type12 = self._interpolate_missing_value(df_type12)
        df_type3 = self._interpolate_missing_value(df_type3)
        # encode
        df_type12["is_type_IorII"] = 1
        df_type3["is_type_IorII"] = 0
        return df_type12.append(df_type3)

def remove_anomalies_by_isolation_forest(X, y):
    """remove outliers with isolation forest

    :param X: DataFrame: train features
    :param y: Series: train target variable
    :return: (DataFrame, Series): cleaned features and target variable
    """
    detector = IsolationForest(random_state=RANSOM_SEED)
    result = detector.fit_predict(X)
    return X.loc[result!=-1, :], y.loc[result!=-1, :]