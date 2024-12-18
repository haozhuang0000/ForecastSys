import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class BBGDataEDA:

    def __init__(self):
        pass

    def check_NAs(self, df_combined, cols_to_process):
        # Generate the statistics of missing na
        for col in cols_to_process:
            nan_count = df_combined[col].isna().sum()
            nan_count_percentage = (nan_count / df_combined.shape[0]) * 100
            print(
                f"The number of NaN values in column '{col}' is: {nan_count} and the percentage is {nan_count_percentage:.1f}%")
