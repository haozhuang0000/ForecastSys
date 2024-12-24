import pmdarima as pm
from autots import AutoTS

class TSModelAuto:

    def __init__(self):
        pass

    def forecast_auto_arima(self, training_points, forecast_points):
        """
        Forecast future values using an automatic ARIMA model.

        Parameters:
        - training_points (array-like): Historical time series data used to train the ARIMA model.
        - forecast_points (int): The number of future data points to forecast.

        Returns:
        - list: A list containing the forecasted values.
        """
        # Fit the Auto ARIMA model
        try:
            # Fit the Auto ARIMA model
            model = pm.auto_arima(
                y=training_points,
                X=None,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True
            )

            # Forecast future points
            forecast, confint = model.predict(n_periods=forecast_points, return_conf_int=True)

            # Convert forecast results to list and return
            return forecast.tolist()

        except Exception as e:
            print(f"ARIMA model failed with error: {e}")
            print("Using the last known value for all forecast points.")

            # Use the last known value for all forecast points
            if len(training_points) > 0:
                last_value = training_points[-1]
            else:
                last_value = 0  # Default to zero if training data is empty

            forecast = [last_value] * forecast_points
            return forecast

    def forecast_AutoTS(self, df, weights, target_var, forecast_points):
        """
        Forecast future values using the AutoTS library, which automates time series modeling.

        Parameters:
        - df (DataFrame): The input dataframe containing the time series data.
        - weights (float): The weight to assign to the target variable during modeling.
        - target_var (str): The name of the target variable in the dataframe.
        - forecast_points (int): The number of future data points to forecast.

        Returns:
        - list: A list containing the forecasted values for the target variable.
        """

        # Handle case where length of df is less than 3
        if len(df) < 3:
            # Use the latest value of the target variable to forecast
            latest_value = df[target_var].iloc[-1]
            return [latest_value] * forecast_points

        # Otherwise, use AutoTS for forecasting
        weights_of_target_var = {target_var: weights}

        model = AutoTS(
            forecast_length=forecast_points,
            frequency='infer',
            prediction_interval=0.9,
            ensemble='auto',
            model_list="superfast",  # "superfast", "default", "fast_parallel"
            transformer_list="superfast",  # "superfast",
            drop_most_recent=1,
            max_generations=4,
            num_validations=2,
            validation_method="backwards",
            verbose=0
        )

        model = model.fit(
            df,
            date_col=None,
            value_col=None,
            id_col=None,
            weights=weights_of_target_var
        )

        prediction = model.predict()

        # Print the details of the best model
        print(model)

        # Point forecasts dataframe
        forecasts_df = prediction.forecast

        return forecasts_df[target_var].tolist()