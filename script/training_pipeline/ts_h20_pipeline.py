import pandas as pd
from tqdm import tqdm
class TSH20Pipeline:

    def __init__(self):
        pass

    def time_series_h20_forecasting_pipeline(self, df, h20_model, categorical_var, features, target):
        """
        Perform time series forecasting with using H2O AutoML.

        Args:
            df (pd.DataFrame): Time series data containing features and target variable.
            h20_model (function): Forecasting function using H2O AutoML.
            categorical_var (list): List of categorical variables to be treated as factors.
            features (list): List of feature column names.
            target (str): Target variable column name.

        Returns:
            tuple: A tuple containing:
                - result (list): List of lists containing forecasted values for each forecasting point.
                - result_flatten (list): Flattened list of all forecasted values.
        """

        training_start_index = 0
        training_end_index = 0
        forecasting_point = 0

        result = []
        end_of_series = False

        for y in tqdm(df[target], total=len(df),
                                desc=f"running pipeline: **{self.time_series_h20_forecasting_pipeline.__name__}** - model: **{model.__name__}** - targetcolumn: **{target}**"):

            if pd.notna(y):
                if end_of_series:

                    # Define the scope of forecasting
                    df_train = df.iloc[training_start_index:training_end_index].reset_index(drop=True)
                    df_test = df.iloc[training_end_index:training_end_index + forecasting_point].reset_index(drop=True)
                    print(df_train)
                    print(df_test)
                    # print('......................')

                    # Begin Forecasting
                    forecast_values = h20_model(df_train, df_test, categorical_var, features, target)
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
        df_train = df.iloc[training_start_index:training_end_index].reset_index(drop=True)
        df_test = df.iloc[training_end_index:training_end_index + forecasting_point].reset_index(drop=True)

        forecast_values = h20_model(df_train, df_test, categorical_var, features, target)
        result.append(forecast_values)

        result_flatten = [item for sublist in result for item in sublist]

        return result, result_flatten