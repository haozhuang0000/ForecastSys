import numpy as np
import statsmodels.api as sm

class TSModelPure:

    def __init__(self, lag: int = 1):
        self.lag = lag

    def forecast_ar(self, training_points, forecast_points):
        """
        Forecast future values using an AR(1) (autoregressive) model.

        Parameters:
        - training_points (list or array-like): Historical time series data used to train the AR(1) model.
        - forecast_points (int): The number of future data points to forecast.

        Returns:
        - list: A list containing the forecasted values.
        """
        # Convert the list of training_pipeline points to a numpy array
        training_points = np.array(training_points)

        # Fit the AR(1) model using statsmodels
        ar_model = sm.tsa.AutoReg(training_points, lags=self.lag).fit()

        # Get the initial training_pipeline series for forecasting
        start_point = len(training_points)
        end_point = start_point + forecast_points - 1

        # Use the fitted AR model to make forecasts
        forecast = ar_model.predict(start=start_point, end=end_point)

        # Convert forecast results to list and return
        return forecast.tolist()

    def forecast_multivariate_ar(self, y_train, X_train, forecast_points, X_test):
        """
        Forecast future values using a multivariate AR(1) model with exogenous variables.

        Parameters:
        - y_train (array-like): The endogenous variable (dependent variable) time series data used for training_pipeline.
        - X_train (array-like): The exogenous variables (independent variables) corresponding to y_train.
        - forecast_points (int): The number of future data points to forecast.
        - X_test (array-like): The exogenous variables for the forecast period.

        Returns:
        - list: A list containing the forecasted values.
        """
        # Fit the AR(n) model using statsmodels with exogenous variables
        ar_model = sm.tsa.AutoReg(endog=y_train, exog=X_train, lags=self.lag).fit()

        # Get the start and end points for the forecast
        start_point = len(y_train)
        end_point = start_point + forecast_points - 1

        # Use the fitted AR model to make forecasts
        forecast = ar_model.predict(start=start_point, end=end_point, exog_oos=X_test)

        # Convert forecast results to list and return
        return forecast.tolist()
