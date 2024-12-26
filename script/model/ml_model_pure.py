import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

class MLModelPure:

    def __init__(self):
        pass

    def rf_regression(self, y_train, X_train, forecast_points, X_test):
        """
        Perform regression using a Random Forest model and forecast future values.

        Parameters:
        - y_train (array-like): The target variable values for training_pipeline.
        - X_train (array-like): The feature variables for training_pipeline.
        - forecast_points (int): Not used in this function but kept for consistency.
        - X_test (array-like): The feature variables for which predictions are to be made.

        Returns:
        - list: A list containing the predicted values.
        """
        # Define the Random Forest model
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            max_features=None,
            random_state=42
        )

        # Define the parameter grid to search over
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10, 15],
            'max_features': ['sqrt', 'log2', None],
        }

        cv = min(5, len(X_train))  # Ensure cv is not greater than the number of samples

        if cv >= 2:
            grid_search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_grid,
                n_iter=20,
                cv=cv,
                n_jobs=-1,
                verbose=False,
                random_state=42,
                scoring='neg_root_mean_squared_error',
            )
            # Fit the model using grid search on training_pipeline data
            grid_search.fit(X_train, y_train)

            # Get the best parameters from grid search
            best_params = grid_search.best_params_
            # print("Best parameters found by grid search:", best_params)

            # Use the best estimator from grid search
            best_rf = grid_search.best_estimator_
        else:
            # Not enough data for cross-validation; fit the model directly
            rf.fit(X_train, y_train)
            best_rf = rf  # Assign the directly fitted model as the best model

        # Make predictions on the test set using the best model
        y_pred = best_rf.predict(X_test)

        return list(y_pred)

    def lightgbm_regression(self, y_train, X_train, forecast_points, X_test):
        """
        Perform regression using a LightGBM model and forecast future values.

        Parameters:
        - y_train (array-like): The target variable values for training_pipeline.
        - X_train (array-like): The feature variables for training_pipeline.
        - forecast_points (int): Not used in this function but kept for consistency.
        - X_test (array-like): The feature variables for which predictions are to be made.

        Returns:
        - list: A list containing the predicted values.
        """
        # Adjusted parameters for regression
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'class_weight': None,
            'min_split_gain': 0.0,
            'min_child_weight': 0.001,
            'min_child_samples': 7,
            'subsample': 1.0,
            'subsample_freq': 0,
            'colsample_bytree': 1.0,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'random_state': None,
            'n_jobs': -1,
            'verbose': -1,
            'device': 'gpu'
        }

        # Create parameters to search
        grid_params = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 500, 1000],
            'num_leaves': [8, 16, 45],
            'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
            'max_depth': [-1, 5, 10, 20],
        }

        # Create the regressor
        mod = lgb.LGBMRegressor(**params)

        # Adjust cv based on the size of X_train
        cv = min(5, len(X_train))  # Ensure cv is not greater than the number of samples

        if cv >= 2:
            grid = RandomizedSearchCV(
                estimator=mod,
                param_distributions=grid_params,
                n_iter=20,
                cv=cv,
                n_jobs=-1,
                verbose=False,
                random_state=42,
                scoring='neg_root_mean_squared_error',
            )
            # Fit the model using grid search on training_pipeline data
            grid.fit(X_train, y_train)

            # Use the best estimator from grid search
            best_model = grid.best_estimator_
        else:
            # Not enough data for cross-validation; fit the model directly
            mod.fit(X_train, y_train)
            best_model = mod  # Assign the directly fitted model as the best model

        # Make predictions on the test set using the best model
        y_pred = best_model.predict(X_test)

        return list(y_pred)

    def rf_regression_with_cat(self, y_train, X_train, forecast_points, X_test, categorical_features=None):
        """
        Perform regression using a Random Forest model and forecast future values,
        handling categorical variables via encoding.

        Parameters:
        - y_train (array-like): The target variable values for training_pipeline.
        - X_train (DataFrame): The feature variables for training_pipeline.
        - forecast_points (int): Not used in this function but kept for consistency.
        - X_test (DataFrame): The feature variables for which predictions are to be made.
        - categorical_features (list, optional): List of column names of categorical features.

        Returns:
        - list: A list containing the predicted values.
        """

        # Ensure that X_train and X_test are pandas DataFrames
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train must be a pandas DataFrame.")
        if not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_test must be a pandas DataFrame.")

        # Identify numerical features
        if categorical_features is not None:
            numerical_features = [col for col in X_train.columns if col not in categorical_features]
        else:
            numerical_features = X_train.columns.tolist()
            categorical_features = []

        # Preprocessing for categorical data
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Define the Random Forest model
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            max_features='auto',
            random_state=42
        )

        # Create a pipeline that first preprocesses the data and then fits the model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', rf)
        ])

        # Define the parameter grid to search over
        param_grid = {
            'regressor__n_estimators': [100, 200, 500],
            'regressor__max_depth': [10, 20, 30, None],
            'regressor__min_samples_split': [2, 5, 10, 15],
            'regressor__max_features': ['sqrt', 'log2', None],
        }

        cv = min(5, len(X_train))  # Ensure cv is not greater than the number of samples

        if cv >= 2:
            grid_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=20,
                cv=cv,
                n_jobs=-1,
                verbose=0,
                random_state=42,
                scoring='neg_root_mean_squared_error',
            )
            # Fit the model using grid search on training_pipeline data
            grid_search.fit(X_train, y_train)

            # Get the best parameters from grid search
            best_params = grid_search.best_params_
            # print("Best parameters found by grid search:", best_params)

            # Use the best estimator from grid search
            best_model = grid_search.best_estimator_
        else:
            # Not enough data for cross-validation; fit the model directly
            model.fit(X_train, y_train)
            best_model = model  # Assign the directly fitted model as the best model

        # Make predictions on the test set using the best model
        y_pred = best_model.predict(X_test)

        return list(y_pred)

    def lightgbm_regression_with_cat(self, y_train, X_train, forecast_points, X_test, categorical_features=None):
        """
        Perform regression using a LightGBM model and forecast future values.

        Parameters:
        - y_train (array-like): The target variable values for training_pipeline.
        - X_train (DataFrame): The feature variables for training_pipeline.
        - forecast_points (int): Not used in this function but kept for consistency.
        - X_test (DataFrame): The feature variables for which predictions are to be made.
        - categorical_features (list, optional): List of column names of categorical features.

        Returns:
        - list: A list containing the predicted values.
        """

        # Adjusted parameters for regression
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'class_weight': None,
            'min_split_gain': 0.0,
            'min_child_weight': 0.001,
            'min_child_samples': 7,
            'subsample': 1.0,
            'subsample_freq': 0,
            'colsample_bytree': 1.0,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'random_state': None,
            'n_jobs': -1,
            'verbose': -1,
            'device': 'gpu'
        }

        # Create parameters to search
        grid_params = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 500, 1000],
            'num_leaves': [8, 16, 45],
            'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
            'max_depth': [-1, 5, 10, 20],
        }

        # Create the regressor
        mod = lgb.LGBMRegressor(**params)

        # Adjust cv based on the size of X_train
        cv = min(5, len(X_train))  # Ensure cv is not greater than the number of samples

        # Ensure that X_train and X_test are pandas DataFrames
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train must be a pandas DataFrame.")
        if not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_test must be a pandas DataFrame.")

        # Set the data type of categorical features to 'category'
        if categorical_features is not None:
            for col in categorical_features:
                X_train[col] = X_train[col].astype('category')
                X_test[col] = X_test[col].astype('category')

        if cv >= 2:
            grid = RandomizedSearchCV(
                estimator=mod,
                param_distributions=grid_params,
                n_iter=20,
                cv=cv,
                n_jobs=-1,
                verbose=False,
                random_state=42,
                scoring='neg_root_mean_squared_error',
            )

            # Fit the model using grid search on training_pipeline data
            grid.fit(X_train, y_train, categorical_feature=categorical_features)

            # Use the best estimator from grid search
            best_model = grid.best_estimator_
        else:
            # Not enough data for cross-validation; fit the model directly
            mod.fit(X_train, y_train, categorical_feature=categorical_features)
            best_model = mod  # Assign the directly fitted model as the best model

        # Make predictions on the test set using the best model
        y_pred = best_model.predict(X_test)

        return list(y_pred)
