from script.training import (TSUnivariatePipeline,
                             TSMultivariatePipeline,
                             TSAutoPipeline,
                             TSMultivariateWLAGPipeline)
from script.model_eval import Metrics, PlotResult
from script.data_variables.variables import Variables
from script.training_helper.training_helper import TrainingHelper
from script.model.ts_model_pure import TSModelPure
from script.model.ts_model_auto import TSModelAuto
from script.model.ml_model_pure import MLModelPure
from script.model.h2o_model_auto import H2OModelAuto
import os
class TrainingMain:

    def __init__(self):
        # ----------------------- pipelines ----------------------- #
        self.ts_univariate_pipeline = TSUnivariatePipeline()
        self.ts_multivariate_pipeline = TSMultivariatePipeline()
        self.ts_auto_pipeline = TSAutoPipeline()
        self.ts_multivariate_wlag_pipeline = TSMultivariateWLAGPipeline()

        # ------------------------- models ------------------------- #
        self.tsmodelpure = TSModelPure()
        self.tsmodelauto = TSModelAuto()
        self.mlmodelpure = MLModelPure()
        self.h2omodelauto = H2OModelAuto()

        self.forecast_ar = self.tsmodelpure.forecast_ar
        self.forecast_auto_arima = self.tsmodelauto.forecast_auto_arima
        self.forecast_autots = self.tsmodelauto.forecast_AutoTS
        self.rf_regression = self.mlmodelpure.rf_regression
        self.lightgbm_regression = self.mlmodelpure.lightgbm_regression
        self.forecast_h20 = self.h2omodelauto.forecast_h20
        self.lightgbm_regression_wcat = self.mlmodelpure.lightgbm_regression_with_cat
        self.rf_regression_wcat = self.mlmodelpure.rf_regression_with_cat

        # ---------------------- model metrics ---------------------- #
        self.metrics = Metrics()
        self.plot_result = PlotResult()

        # ------------------------- data ------------------------- #
        self.DATA_PATH = "../../data/output/union_processed_imputed_80_20.csv"
        self.DATA_PATH_GROUDTRUTH = "../../data/output/union_processed_groundtruth_80_20.csv"

        self.variables = Variables()
        self.x_cols_to_forecast = self.variables.X_SELECTED
        self.industry_info_cols = self.variables.INDUSTRY_TIC_INFO
        self.y = self.variables.Y_SELECTED

        # ------------------------- helper ------------------------- #
        self.helper = TrainingHelper(self.variables)

        # ------------------------- results ------------------------- #
        self.result_map = {}
        self.result_ar1_map = {}
        self.result_arima_map = {}

        self.result_rf_ar1_map = {}
        self.result_rf_arima_map = {}

        self.result_lightgbm_ar1_map = {}
        self.result_lightgbm_arima_map = {}
        self.result_rf_lag_map = {}
        self.result_lightgbm_lag_map = {}

        self.result_auto_ts_map = {}

        self.result_h20_industry_ar1_map = {}
        self.result_h20_industry_arima_map = {}

        self.result_lightgbm_industry_ar1_map = {}
        self.result_lightgbm_industry_arima_map = {}

        self.result_rf_industry_ar1_map = {}
        self.result_rf_industry_arima_map = {}

    def _run_ts_univariate_pipeline(self, df, df_ground_truth,
                                    model, result_metrics_map):

        for y in self.y:
            result, result_flatten, training_data_with_forecast_result = self.ts_univariate_pipeline.time_series_forecasting_pipeline(
                    df, y, model
                )

            metrics = self.metrics.calculate_metrics(result_flatten, df_ground_truth[y])
            result_metrics_map[y] = metrics

    def _run_ts_multivariate_pipeline(self, df, df_ground_truth, df_x_ar_one,
                                     model, result_metrics_map):

        for y in self.y:
            df_y = df[y]
            df_x_ar_one_drop = df_x_ar_one.drop(columns=[y])
            result, result_flatten = self.ts_multivariate_pipeline.time_series_multi_forecasting_pipeline(
                df_x_ar_one_drop, df_y, model
            )
            metrics = self.metrics.calculate_metrics(result_flatten, df_ground_truth[y])
            result_metrics_map[y] = metrics

    def _run_ts_multivariate_wlag_pipeline(self, df, df_ground_truth, df_x,
                                     model, result_metrics_map):

        for y in self.y:
            df_y = df[y]
            df_x_drop = df_x.drop(columns=[y])
            x_col = [x for x in self.x_cols_to_forecast if x != y]
            result, result_flatten = self.ts_multivariate_wlag_pipeline.time_series_multi_with_lag_forecasting_pipeline(
                df_x_drop, df_y, model, x_col, y
            )
            metrics = self.metrics.calculate_metrics(result_flatten, df_ground_truth[y])
            result_metrics_map[y] = metrics

    def _run_ts_auto_pipeline(self, df, df_ground_truth,
                              model, result_metrics_map):
        df_auto_ts = df.set_index("Year")
        df_auto_ts.drop(columns=['TICKER'], inplace=True)

        for y in self.y:
            result, result_flatten = self.ts_auto_pipeline.time_series_AutoTS_forecasting_pipeline(
            df_auto_ts, model, 30, y
            )
            metrics = self.metrics.calculate_metrics(result_flatten, df_ground_truth[y])
            result_metrics_map[y] = metrics

    def _run_h2o_pipeline(self, df_industry_train, df_industry_test, df_ground_truth, result_metrics_map):

        for y in self.y:
            df_train = df_industry_train[self.x_cols_to_forecast + self.industry_info_cols]
            df_test = df_industry_test[self.x_cols_to_forecast + self.industry_info_cols]

            # TODO: format it as pipeline
            result_h20_industry_ar1 = self.h2omodelauto.forecast_h20(df_train, df_test, self.industry_info_cols,
                                                   self.x_cols_to_forecast + self.industry_info_cols, y)

            metrics = self.metrics.calculate_metrics(result_h20_industry_ar1, df_ground_truth[y])
            result_metrics_map[y] = metrics

    def _run_ml_industry_pipeline(self, df_industry_train, df_industry_test, df_ground_truth, model, result_metrics_map):

        for y in self.y:
            df_train = df_industry_train[self.x_cols_to_forecast + self.industry_info_cols]
            df_test = df_industry_test[self.x_cols_to_forecast + self.industry_info_cols]
            y_train = df_industry_train[y]

            # TODO: format it as pipeline
            result_flatten = model(y_train, df_train, 0, df_test, self.industry_info_cols)
            metrics = self.metrics.calculate_metrics(result_flatten, df_ground_truth[y])
            result_metrics_map[y] = metrics
    def _run_plot_result(self):

        for y in self.y:
            models = ['ARIMA', 'Random Forest with X:AR(1)', 'Random Forest with X:ARIMA', 'Random Forest with Lag',
                      'Light GBM wiwth X:AR(1)', 'Light GBM wiwth X:ARIMA', 'Light GBM with Lag', 'AutoTS',
                      'H20 AR1 with industry', 'H20 ARIMA with industry', 'LightGBM AR1 with industry',
                      'LightGBM ARIMA with industry', 'RF AR1 with industry', 'RF ARIMA with industry']
            result = [self.result_arima_map, self.result_rf_ar1_map, self.result_rf_arima_map, self.result_rf_lag_map,
                      self.result_lightgbm_ar1_map, self.result_lightgbm_arima_map, self.result_lightgbm_lag_map, self.result_auto_ts_map,
                      self.result_h20_industry_ar1_map, self.result_h20_industry_arima_map, self.result_lightgbm_industry_ar1_map,
                      self.result_lightgbm_industry_arima_map, self.result_rf_industry_ar1_map, self.result_rf_industry_arima_map]

            self.plot_result.plot_result(y, models, result)

    def save_results_to_json(self, output_dir="results"):
        """
        Save all result maps as JSON files in the specified directory.
        """
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save each result map to a JSON file
        for attr_name in dir(self):
            if attr_name.startswith("result_") and isinstance(getattr(self, attr_name), dict):
                file_path = os.path.join(output_dir, f"{attr_name}.json")
                with open(file_path, "w") as json_file:
                    json.dump(getattr(self, attr_name), json_file, indent=4)
                print(f"Saved {attr_name} to {file_path}")

    def main(self):

        ### --------------------------- Loading training and test data ------------------------ ###
        df, df_x, df_ground_truth = self.helper.read_data(self.DATA_PATH, self.DATA_PATH_GROUDTRUTH)
        df_industry_train, df_test, nan_rows = self.helper._prepare_train_test(df)
        df_x_ar_one, df_industry_ar_test = self.helper._prepare_industry_ar(df_x, df_test, nan_rows, self.forecast_ar)
        df_x_arima_one, df_industry_arima_test = self.helper._prepare_industry_arima(df_x, df_test, nan_rows, self.forecast_auto_arima)

        ### --------------------------- run univariate ts model ------------------------ ###
        self._run_ts_univariate_pipeline(df, df_ground_truth, self.forecast_ar, self.result_ar1_map)
        self._run_ts_univariate_pipeline(df, df_ground_truth, self.forecast_auto_arima, self.result_arima_map)

        ### --------------------------- run multi-variate ts model ------------------------ ###

        # RF regression with forecast X:AR(1)
        self._run_ts_multivariate_pipeline(df, df_ground_truth, df_x_ar_one, self.rf_regression, self.result_rf_ar1_map)

        # RF regression with forecast X:ARIMA
        self._run_ts_multivariate_pipeline(df, df_ground_truth, df_x_arima_one, self.rf_regression, self.result_rf_arima_map)

        # Lightgbm with forecast X:AR(1)
        self._run_ts_multivariate_pipeline(df, df_ground_truth, df_x_ar_one, self.lightgbm_regression, self.result_lightgbm_ar1_map)

        # Lightgbm with forecast X:ARIMA
        self._run_ts_multivariate_pipeline(df, df_ground_truth, df_x_arima_one, self.lightgbm_regression, self.result_lightgbm_arima_map)

        ### --------------------------- run multi-variate ts model w/ lag ------------------------ ###

        # RF Lag Methodology forecast
        self._run_ts_multivariate_wlag_pipeline(df, df_ground_truth, df_x, self.rf_regression, self.result_rf_lag_map)

        # LightGBM Lag Methodology forecast
        self._run_ts_multivariate_wlag_pipeline(df, df_ground_truth, df_x, self.lightgbm_regression, self.result_lightgbm_lag_map)

        ### --------------------------- run auto ts ------------------------ ###

        # Auto Ts
        self._run_ts_auto_pipeline(df, df_ground_truth, self.forecast_autots, self.result_auto_ts_map)

        ### --------------------------- run H20 ------------------------ ###

        # H20 AR1 with industry
        self._run_h2o_pipeline(df_industry_train, df_industry_ar_test, df_ground_truth, self.result_h20_industry_ar1_map)

        # H20 ARIMA with industry
        self._run_h2o_pipeline(df_industry_train, df_industry_arima_test, df_ground_truth, self.result_h20_industry_arima_map)

        ### --------------------------- run ml with industry ------------------------ ###

        # LightBGM AR1 with industry
        self._run_ml_industry_pipeline(df_industry_train, df_industry_ar_test, df_ground_truth, self.lightgbm_regression_wcat, self.result_lightgbm_industry_ar1_map)

        # LightBGM ARIMA with industry
        self._run_ml_industry_pipeline(df_industry_train, df_industry_arima_test, df_ground_truth, self.lightgbm_regression_wcat, self.result_lightgbm_industry_arima_map)

        # RF AR1 with industry
        self._run_ml_industry_pipeline(df_industry_train, df_industry_ar_test, df_ground_truth, self.rf_regression_wcat, self.result_rf_industry_ar1_map)

        # RF ARIMA with industry
        self._run_ml_industry_pipeline(df_industry_train, df_industry_arima_test, df_ground_truth, self.rf_regression_wcat, self.result_rf_industry_arima_map)

        self._run_plot_result()
        self.save_results_to_json("../../data/results")


if __name__ == "__main__":
    training_main = TrainingMain()
    training_main.main()