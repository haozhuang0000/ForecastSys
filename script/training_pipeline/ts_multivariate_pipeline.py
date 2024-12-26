import pandas as pd
from tqdm import tqdm
class TSMultivariatePipeline:

    def __init__(self):
        pass

    def time_series_multi_forecasting_pipeline(self, df_train_x, df_train_y, model, target_column):
        """
        Processes multivariate time series data, segments it into individual series,
        and applies a forecasting model to each series using exogenous variables.

        Parameters:
        - df_train_x (DataFrame): The input DataFrame containing the exogenous variables (features).
        - df_train_y (Series or DataFrame): The target variable time series data.
        - model (function): A forecasting function that takes in y_train, X_train, forecast_points, and X_test.

        Returns:
        - result (list): A list of forecasted values for each series.
        - result_flatten (list): A flattened list of all forecasted values.
        """
        training_start_index = 0
        training_end_index = 0
        forecasting_point = 0

        result = []
        end_of_series = False

        for y in tqdm(df_train_y, total=len(df_train_y),
                                desc=f"running pipeline: **{self.time_series_multi_forecasting_pipeline.__name__}** - model: **{model.__name__}** - targetcolumn: **{target_column}**"):

            if pd.notna(y):
                if end_of_series:

                    # Define the scope of forecasting
                    y_train = df_train_y.iloc[training_start_index:training_end_index].reset_index(drop=True)
                    X_train = df_train_x.iloc[training_start_index:training_end_index].reset_index(drop=True)
                    X_test = df_train_x.iloc[training_end_index:training_end_index + forecasting_point].reset_index(
                        drop=True)

                    # Begin Forecasting
                    forecast_values = model(y_train, X_train, forecasting_point, X_test)
                    result.append(forecast_values)

                    end_of_series = False

                    training_start_index, training_end_index = training_end_index + forecasting_point, training_end_index + forecasting_point + 1
                    forecasting_point = 0
                else:
                    training_end_index += 1
            else:
                forecasting_point += 1
                end_of_series = True

        # For the final timeseries
        y_train = df_train_y.iloc[training_start_index:training_end_index].reset_index(drop=True)
        X_train = df_train_x.iloc[training_start_index:training_end_index].reset_index(drop=True)
        X_test = df_train_x.iloc[training_end_index:training_end_index + forecasting_point].reset_index(drop=True)

        forecast_values = model(y_train, X_train, forecasting_point, X_test)
        result.append(forecast_values)

        result_flatten = [item for sublist in result for item in sublist]

        return result, result_flatten