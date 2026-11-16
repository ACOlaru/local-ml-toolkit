"""Dataset validation utilities."""

import numpy as np
import pandas as pd


class DatasetValidator:
    """Check datasets for common quality problems."""

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def check_empty(self):
        return self.dataframe.empty

    def check_missing_values(self):
        return self.dataframe.isnull().sum().to_dict()

    def check_duplicates(self):
        return int(self.dataframe.duplicated().sum())

    def check_constant_columns(self):
        return [
            column
            for column in self.dataframe.columns
            if self.dataframe[column].nunique(dropna=False) <= 1
        ]

    def check_numeric_columns(self):
        return self.dataframe.select_dtypes(
            include=np.number
        ).columns.tolist()

    def check_infinite_values(self):
        numeric = self.dataframe.select_dtypes(include=np.number)
        return {
            column: int(np.isinf(numeric[column]).sum())
            for column in numeric.columns
        }

    def generate_report(self):
        return {
            "empty": self.check_empty(),
            "missing_values": self.check_missing_values(),
            "duplicates": self.check_duplicates(),
            "constant_columns": self.check_constant_columns(),
            "numeric_columns": self.check_numeric_columns(),
            "infinite_values": self.check_infinite_values(),
        }