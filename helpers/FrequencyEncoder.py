import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Custom transformer that frequency-encodes the specified categorical columns.
    Optionally drops the original columns.

    Parameters:
    - columns: List of column names to frequency encode
    - drop_original: Whether to drop the original columns (default: False)
    """
    def __init__(self, columns=None, drop_original=False):
        self.columns = columns
        self.drop_original = drop_original
        self.freq_maps_ = {}

    def fit(self, X, y=None):
        if self.columns is None:
            raise ValueError("`columns` must be specified before fitting.")

        X = X.copy()
        for col in self.columns:
            # Map on string representations to ensure safety
            self.freq_maps_[col] = (
                X[col].astype(str).value_counts(normalize=True).to_dict()
            )
        return self

    def transform(self, X):
        X_out = X.copy()

        for col in self.columns:
            col_str = X_out[col].astype(str)
            freq_col = f"{col}_freq"
            X_out[freq_col] = col_str.map(self.freq_maps_[col]).fillna(0.0).astype(float)

        if self.drop_original:
            X_out.drop(columns=self.columns, inplace=True)

        return X_out
