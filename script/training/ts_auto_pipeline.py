import pandas as pd

class TSAutoPipeline:

    def __init__(self):
        pass

    def time_series_AutoTS_forecasting_pipeline(self, df, forecast_AutoTS, weights, target_var):
        """
        Processes a DataFrame containing time series data, segments it into individual series,
        and applies the AutoTS forecasting model to each series.

        Parameters:
        - df (DataFrame): The input DataFrame containing the time series data.
        - weights (float or dict): The weight(s) to assign to the target variable during modeling.
        - target_var (str): The name of the target variable in the DataFrame.

        Returns:
        - result (list): A list of forecasted values for each series.
        - result_flatten (list): A flattened list of all forecasted values.
        """
        training_start_index = 0
        training_end_index = 0
        forecasting_point = 0

        result = []
        end_of_series = False

        for index, row in df.iterrows():

            if pd.notna(row[target_var]):
                if end_of_series:

                    # Define the scope of forecasting
                    df_main = df.iloc[training_start_index:training_end_index]

                    # Begin Forecasting
                    forecast_values = forecast_AutoTS(df_main, weights, target_var, forecasting_point)
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
        df_main = df.iloc[training_start_index:training_end_index]
        forecast_values = forecast_AutoTS(df_main, weights, target_var, forecasting_point)
        result.append(forecast_values)

        result_flatten = [item for sublist in result for item in sublist]

        return result, result_flatten