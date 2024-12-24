import h2o
from h2o.automl import H2OAutoML

class H2OModelAuto:

    def __init__(self):
        h2o.init()

    def forecast_h20(self, df_train, df_test, categorical_var, features, target):
        """
        Perform forecasting using H2O AutoML.

        Args:
            df_train (pd.DataFrame): Training dataset containing features and target variable.
            df_test (pd.DataFrame): Test dataset containing features for which predictions are required.
            categorical_var (list): List of categorical variables to be treated as factors.
            features (list): List of feature column names.
            target (str): Target variable column name.

        Returns:
            list: Predicted values for the test dataset.
        """

        df_train_h20 = h2o.H2OFrame(df_train)
        df_test_h20 = h2o.H2OFrame(df_test)

        for var in categorical_var:
            df_train_h20[var] = df_train_h20[var].asfactor()
            df_test_h20[var] = df_test_h20[var].asfactor()

        df_test_h20 = df_test_h20.drop(target)

        aml = H2OAutoML(
            max_models=20,
            seed=1,
            exclude_algos=["DeepLearning"],  # Optionally exclude algorithms
            sort_metric="RMSE"  # Metric to sort models
        )

        aml.train(x=features, y=target, training_frame=df_train_h20)

        # Check if any models were built
        if aml.leader is not None:
            best_model = aml.leader
            predictions = best_model.predict(df_test_h20)
            predictions_df = predictions.as_data_frame()
            predictions_list = predictions_df['predict'].tolist()
        else:
            # Predict zeros for the forecasting points
            last_value = df_train[target].iloc[-1]
            predictions_list = [last_value] * df_test.shape[0]

        return predictions_list