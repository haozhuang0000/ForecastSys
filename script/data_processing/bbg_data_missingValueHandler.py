import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from script.logger.logger import Log
class BBGDataMissingValueHandler:

    def __init__(self):
        self.logger = Log(f"{os.path.basename(__file__)}").getlog()
        pass

    def mark_high_missing_columns(self, df, threshold=0.8):
        """
        Replaces columns where more than 'threshold' fraction of values are missing
        with np.nan.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            threshold (float): The fraction of allowed missing values. Columns with
                               a higher fraction of missing values will be replaced
                               with np.nan.

        Returns:
            pd.DataFrame: The DataFrame with specified columns replaced by np.nan.
        """
        # Calculate the fraction of missing values in each column
        missing_fraction = df.isna().mean()

        # Identify columns where the missing fraction exceeds the threshold
        cols_to_mark = missing_fraction[missing_fraction > threshold].index

        # Replace entire columns with np.nan
        df[cols_to_mark] = 0

        return df

    def handle_missing_start(self, series):
        """
        Handles missing values at the start of the series using backfill.
        """
        first_valid_idx = series.first_valid_index()
        if first_valid_idx is not None and first_valid_idx != series.index[0]:
            # Get the positional index of the first valid index
            first_valid_pos = series.index.get_loc(first_valid_idx)
            # Backfill missing values at the start
            series.iloc[:first_valid_pos] = series.iloc[first_valid_pos]
        return series

    def handle_missing_end(self, series):
        """
        Handles missing values at the end of the series using linear regression
        based on the last continuous block of valid data, with X_train starting from 0.
        """
        last_valid_idx = series.last_valid_index()
        if last_valid_idx is not None and last_valid_idx != series.index[-1]:
            # Get the positional index of the last valid index
            last_valid_pos = series.index.get_loc(last_valid_idx)

            # Find the start of the last continuous block of valid data
            start_pos = last_valid_pos
            while start_pos > 0 and not pd.isna(series.iloc[start_pos - 1]):
                start_pos -= 1

            # Extract the last continuous block of valid data
            y_train = series.iloc[start_pos:last_valid_pos + 1].values

            # Adjust X_train to start from 0
            X_train = np.arange(0, len(y_train)).reshape(-1, 1)

            # Check if we have enough data to train
            if len(y_train) > 1:
                # Build linear regression model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Number of missing values at the end
                num_missing = len(series) - (last_valid_pos + 1)

                # Prepare X_pred starting from where X_train left off
                X_pred = np.arange(len(y_train), len(y_train) + num_missing).reshape(-1, 1)

                # Predict missing values at the end
                y_pred = model.predict(X_pred)

                # Fill missing values
                series.iloc[last_valid_pos + 1:] = y_pred
            else:
                # Not enough data to train a model; use the last valid value
                series.iloc[last_valid_pos + 1:] = series.iloc[last_valid_pos]
        return series

    def handle_missing_between(self, series):
        """
        Handles consecutive missing values between two data points using interpolation.
        """
        if series.dtype not in [float, int, 'int64']:
            # print(f"skipping - {series.name} - dtype: {series.dtype}")
            return series
        series = pd.to_numeric(series)
        series.interpolate(method='linear', inplace=True)
        return series

    def impute_series(self, series):
        """
        Imputes missing values in a pandas Series according to the specified rules.
        """
        series = self.handle_missing_start(series)
        series = self.handle_missing_end(series)
        series = self.handle_missing_between(series)
        return series

    def impute_missing_values(self, df):
        """
        Applies the imputation process to each column in the DataFrame.
        """
        for col in df.columns:
            df[col] = self.impute_series(df[col])
        return df