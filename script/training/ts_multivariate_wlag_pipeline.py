import pandas as pd
import numpy as np

class TSMultivariateWLAGPipeline:

    def __init__(self):
        pass

    @staticmethod
    def mean_at_each_index(input_lists):
        """
        Calculates the mean of elements at each index across multiple lists,
        handling lists of varying lengths.

        Parameters:
        - input_lists (list of lists): A list containing multiple lists of numerical values.

        Returns:
        - mean_list (list): A list containing the mean of elements at each index,
          ignoring None values resulting from padding.
        """
        # Step 1: Determine the maximum length of the lists
        max_length = max(len(lst) for lst in input_lists)

        # Step 2: Pad lists with None to handle varying lengths
        padded_lists = [lst + [None] * (max_length - len(lst)) for lst in input_lists]

        # Step 3: Calculate mean at each index, ignoring None values
        mean_list = [np.nanmean([x for x in col if x is not None]) for col in zip(*padded_lists)]

        return mean_list

    @staticmethod
    def forecast_df_with_lag(df, forecasting_point, lag, model, x_cols_to_forecast, target_col):
        """
        Prepares training and testing datasets with a specified lag and applies a forecasting model.

        Parameters:
        - df (DataFrame): The input DataFrame containing the features and target variable.
        - forecasting_point (int): The number of future data points to forecast.
        - lag (int): The lag value to shift the target variable for training.
        - model (function): A forecasting function that takes in y_train, X_train, forecast_points, and X_test.
        - x_cols_to_forecast (list): A list of column names to be used as features.
        - target_col (str): The name of the target variable column in the DataFrame.

        Returns:
        - forecast_values (list): A list containing the forecasted values.
        """

        result = []

        num_of_rows = df.shape[0]
        X_train = df[x_cols_to_forecast].iloc[:num_of_rows - forecasting_point - lag].reset_index(drop=True)
        y_train = df[target_col].iloc[:num_of_rows - forecasting_point - lag].reset_index(drop=True)
        X_test = df[x_cols_to_forecast].iloc[X_train.shape[0]: num_of_rows - forecasting_point].reset_index(drop=True)
        forecast_values = model(y_train, X_train, forecasting_point, X_test)
        result.append(forecast_values)

        return forecast_values

    def time_series_multi_with_lag_forecasting_pipeline(self, df_train_x, df_train_y, model, x_cols_to_forecast, target_col):
        """
        Processes multivariate time series data with varying lags, segments it into individual series,
        and applies a forecasting model to each series.

        Parameters:
        - df_train_x (DataFrame): The input DataFrame containing the exogenous variables (features).
        - df_train_y (Series or DataFrame): The target variable time series data.
        - model (function): A forecasting function that takes in y_train, X_train, forecast_points, and X_test.
        - x_cols_to_forecast (list): A list of column names to be used as features.
        - target_col (str): The name of the target variable column in the DataFrame.

        Returns:
        - result (list): A list of forecasted values for each series.
        - result_flatten (list): A flattened list of all forecasted values.
        """
        start_index = 0
        end_index = 0
        forecasting_point = 0

        result = []
        end_of_series = False

        for y in df_train_y:

            if pd.notna(y):
                if end_of_series:
                    df_result = []
                    # Define the scope of forecasting
                    y_train = df_train_y.iloc[start_index:end_index + forecasting_point].reset_index(drop=True)
                    X_train = df_train_x.iloc[start_index:end_index + forecasting_point].reset_index(drop=True)

                    # Begin Forecasting at different lag
                    for lag in range(1, forecasting_point + 1):
                        y_train_shift = y_train.shift(-lag)
                        df_concat = pd.concat([y_train_shift, X_train], axis=1).reset_index(drop=True)
                        forecast_value = self.forecast_df_with_lag(df_concat, forecasting_point, lag, model,
                                                              x_cols_to_forecast, target_col)
                        # print(forecast_value)
                        df_result.append(forecast_value)
                    result.append(self.mean_at_each_index(df_result))

                    end_of_series = False
                    start_index, end_index = end_index + forecasting_point, end_index + forecasting_point + 1
                    forecasting_point = 0
                else:
                    end_index += 1
            else:
                forecasting_point += 1
                end_of_series = True

        # For the final timeseries
        df_result = []
        y_train = df_train_y.iloc[start_index:end_index + forecasting_point].reset_index(drop=True)
        X_train = df_train_x.iloc[start_index:end_index + forecasting_point].reset_index(drop=True)

        # Begin Forecasting at different lag
        for lag in range(1, forecasting_point + 1):
            y_train_shift = y_train.shift(-lag)
            df_concat = pd.concat([y_train_shift, X_train], axis=1).reset_index(drop=True)
            forecast_value = self.forecast_df_with_lag(df_concat, forecasting_point, lag, model, x_cols_to_forecast,
                                                  target_col)
            df_result.append(forecast_value)
        result.append(self.mean_at_each_index(df_result))

        result_flatten = [item for sublist in result for item in sublist]

        return result, result_flatten