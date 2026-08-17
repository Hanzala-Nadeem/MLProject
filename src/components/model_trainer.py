import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(random_state=42, verbosity=0),
                "CatBoosting Regressor": CatBoostRegressor(
                    verbose=False,
                    random_seed=42,
                    allow_writing_files=False,
                ),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
                "KNeighbors Regressor": KNeighborsRegressor(),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "absolute_error", "poisson"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "subsample": [0.8, 1.0],
                    "n_estimators": [100, 200],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [100, 200],
                    "max_depth": [3, 6],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [100, 200],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "n_estimators": [50, 100, 200],
                },
                "KNeighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                },
            }

            model_report = {}
            best_estimators = {}

            for model_name, model in models.items():
                logging.info(f"Tuning model: {model_name}")
                param_grid = params.get(model_name, {})

                gs_n_jobs = 1 if model_name == "CatBoosting Regressor" else -1

                if param_grid:
                    gs = GridSearchCV(
                        estimator=model,
                        param_grid=param_grid,
                        cv=3,
                        scoring="r2",
                        n_jobs=gs_n_jobs,
                        error_score="raise",
                    )
                    gs.fit(X_train, y_train)
                    best_model = gs.best_estimator_
                else:
                    model.fit(X_train, y_train)
                    best_model = model

                y_pred = best_model.predict(X_test)
                test_r2 = r2_score(y_test, y_pred)

                model_report[model_name] = test_r2
                best_estimators[model_name] = best_model

                logging.info(f"{model_name} test R2: {test_r2:.4f}")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = best_estimators[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info(f"Best model: {best_model_name} | R2: {best_model_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)