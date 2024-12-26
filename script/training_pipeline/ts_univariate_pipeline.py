import pandas as pd
from tqdm import tqdm

class TSUnivariatePipeline:

    def __init__(self):
        pass


    def time_series_forecasting_pipeline(self, df, target_column, model):
        """
        Processes a DataFrame containing time series data, segments it into individual series,
        and applies a forecasting model to each series.

        Parameters:
        - df (DataFrame): The input DataFrame containing the time series data.
        - target_column (str): The name of the target column in the DataFrame to forecast.
        - model (function): A forecasting function that takes in training_pipeline data and the number of points to forecast.

        Returns:
        - result (list): A list of forecasted values for each series.
        - result_flatten (list): A flattened list of all forecasted values.
        - training_data_with_forecast_result (list): A combined list of training_pipeline data and forecasted values for all series.
        """
        training_data = []
        forecasting_point = 0
        result = []
        training_data_with_forecast_result = []
        end_of_series = False
        for index, row in tqdm(df.iterrows(), total=len(df),
                                desc=f"running pipeline: **{self.time_series_forecasting_pipeline.__name__}** - model: **{model.__name__}** - targetcolumn: **{target_column}**"):
            if pd.notna(row[target_column]):
                if end_of_series and len(training_data) > 3:
                    forecast_values = model(training_data, forecasting_point)
                    result.append(forecast_values)
                    training_data_with_forecast_result.append(training_data)
                    training_data_with_forecast_result.append(forecast_values)
                    training_data = []
                    forecasting_point = 0
                    end_of_series = False


                elif end_of_series and len(training_data) <= 3:

                    forecast_values = [training_data[-1] for _ in range(forecasting_point)]
                    result.append(forecast_values)
                    training_data_with_forecast_result.append(training_data)
                    training_data_with_forecast_result.append(forecast_values)
                    training_data = []
                    forecasting_point = 0
                    end_of_series = False
                training_data.append(row[target_column])
            else:
                forecasting_point += 1
                end_of_series = True

        # For the final timeseries
        forecast_values = model(training_data, forecasting_point)
        result.append(forecast_values)
        training_data_with_forecast_result.append(training_data)
        training_data_with_forecast_result.append(forecast_values)

        result_flatten = [item for sublist in result for item in sublist]
        training_data_with_forecast_result = [item for sublist in training_data_with_forecast_result for item in sublist]

        return result, result_flatten, training_data_with_forecast_result