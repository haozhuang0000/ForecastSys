import pandas as pd
import os
from script.training_pipeline.ts_univariate_pipeline import TSUnivariatePipeline
from script.data_variables.variables import Variables

class TrainingHelper:

    def __init__(self, variables: Variables):
        self.variables = variables
        self.x_cols_to_forecast = self.variables.X_SELECTED
        self.industry_info_cols = self.variables.INDUSTRY_TIC_INFO

    def read_data(self, filepath, filepath_groudtruth):
        df = pd.read_csv(filepath)
        df['Year'] = pd.to_datetime(df['Year'], format='%Y')
        df_x = df[self.x_cols_to_forecast]

        df_ground_truth = pd.read_csv(filepath_groudtruth)
        return df, df_x, df_ground_truth

    def _prepare_train_test(self, df):

        nan_rows = df[df.isnull().any(axis=1)].index.tolist()
        df_industry_train = df.drop(nan_rows)
        df_test = df.loc[nan_rows]

        return df_industry_train, df_test, nan_rows


    def forecast_x_col(self, df_x, x_cols_to_forecast, univariate_model):
        """
        Forecast multiple exogenous variables (features) using a specified univariate time series model, particulary using the AR(1) or Auto Arima

        Parameters:
        - df_x (DataFrame): The input DataFrame containing the time series data for exogenous variables.
        - x_cols_to_forecast (list): A list of column names in df_x to be forecasted.
        - univariate_model (function): A univariate forecasting function (e.g., AR(1), ARIMA) that accepts
          the time series data and the number of points to forecast.

        Returns:
        - df_forecast_x (DataFrame): A DataFrame containing the forecasted values for each specified column,
          combined with the original training_pipeline data.
        """
        forecast_results = {}

        for col in x_cols_to_forecast:
            if col in df_x.columns:
                result, result_flatten, training_data_with_forecast_result = (
                    TSUnivariatePipeline().time_series_forecasting_pipeline(df_x, col, univariate_model))
                forecast_results[col] = training_data_with_forecast_result

        df_forecast_x = pd.DataFrame(forecast_results)

        return df_forecast_x

    def _prepare_industry_ar(self, df_x, df_test, nan_rows, model):

        file_path = '../../data/df_x_ar1.csv'
        if not os.path.exists(file_path):
            df_x_ar_one = self.forecast_x_col(df_x, self.x_cols_to_forecast, model)
            df_x_ar_one.to_csv(file_path, index=False)
        else:
            df_x_ar_one = pd.read_csv(file_path)

        df_industry_ar_test = df_x_ar_one.loc[nan_rows]
        df_industry_ar_test[self.industry_info_cols] = df_test[self.industry_info_cols]
        df_industry_ar_test = df_industry_ar_test.reset_index(drop=True)

        return df_x_ar_one, df_industry_ar_test

    def _prepare_industry_arima(self, df_x, df_test, nan_rows, model):

        file_path = '../../data/df_x_arima.csv'
        if not os.path.exists(file_path):
            df_x_arima = self.forecast_x_col(df_x, self.x_cols_to_forecast, model)
            df_x_arima.to_csv(file_path, index=False)
        else:
            df_x_arima = pd.read_csv(file_path)

        df_industry_arima_test = df_x_arima.loc[nan_rows]
        df_industry_arima_test[self.industry_info_cols] = df_test[self.industry_info_cols]
        df_industry_arima_test = df_industry_arima_test.reset_index(drop=True)

        return df_x_arima, df_industry_arima_test