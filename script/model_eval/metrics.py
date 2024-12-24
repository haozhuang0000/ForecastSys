import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

class Metrics:

    def __init__(self):
        pass

    def rmspe(self, y_true, y_pred):
        # Exclude zero actual sales values to avoid division by zero
        non_zero_mask = y_true != 0
        y_true_filtered = y_true[non_zero_mask]
        y_pred_filtered = y_pred[non_zero_mask]

        # Calculate RMSPE
        return np.sqrt(np.mean(((y_true_filtered - y_pred_filtered) / y_true_filtered) ** 2))

    def calculate_metrics(self, predicted, actual):
        """
        Calculates various evaluation metrics for regression models, including MAE, MAPE, RMSE, RMSPE,
        RMSE standard deviation, RMSE confidence interval, and R² score.

        Parameters:
        - predicted (array-like): The predicted values from the regression model.
        - actual (array-like): The actual observed values.

        Returns:
        - metrics (dict): A dictionary containing the calculated metrics:
            - "MAE": Mean Absolute Error
            - "MAPE": Mean Absolute Percentage Error
            - "RMSE": Root Mean Squared Error
            - "RMSE Std Dev": Standard deviation of RMSE from bootstrapping
            - "RMSE 95% CI": 95% confidence interval for RMSE
            - "RMSPE": Root Mean Squared Percentage Error
            - "R² Score": Coefficient of determination
        """
        y_pred = np.array(predicted)
        y_true = np.array(actual)

        mask = ~np.isnan(y_true)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # 1. MAE (Mean Absolute Error)
        mae = mean_absolute_error(y_true, y_pred)

        # 2. MAPE (Mean Absolute Percentage Error)
        # MAPE = (100/n) * Σ(|(y_true - y_pred) / y_true|)
        epsilon = 1e-10
        y_safe = np.where(y_true == 0, epsilon, y_true)
        mape = np.mean(np.abs((y_true - y_pred) / y_safe)) * 100

        # 3. RMSE (Root Mean Squared Error)
        rmse = root_mean_squared_error(y_true, y_pred)

        # 4. RMSPE
        rmspe_value = self.rmspe(y_true, y_pred)

        # Step 2: Bootstrapping
        n_iterations = 1000
        rmse_values = []

        # Randomly resample and compute RMSE
        for _ in range(n_iterations):
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            actual_sample = y_true[indices]
            predicted_sample = y_pred[indices]
            rmse_sample = root_mean_squared_error(actual_sample, predicted_sample)
            rmse_values.append(rmse_sample)

        # Step 3: Calculate Standard Deviation and Confidence Interval
        rmse_std = np.std(rmse_values)
        lower_ci = np.percentile(rmse_values, 2.5)
        upper_ci = np.percentile(rmse_values, 97.5)

        print(f"RMSE: {rmse}")
        print(f"Standard Deviation of RMSE: {rmse_std}")
        print(f"95% Confidence Interval of RMSE: [{lower_ci}, {upper_ci}]")

        # 4. R² Score
        r2 = r2_score(y_true, y_pred)

        # Return all metrics as a dictionary
        metrics = {
            "MAE": mae,
            "MAPE": mape,
            "RMSE": rmse,
            "RMSE Std Dev": rmse_std,
            "RMSE 95% CI": (lower_ci, upper_ci),
            "RMSPE": rmspe_value,
            "R² Score": r2
        }

        return metrics