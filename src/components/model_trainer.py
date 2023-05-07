import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_models


@dataclass
class ModelTrainerConfig:
    """
    ModelTrainerConfig class is used to store the configuration of the model trainer.
    """
    trained_model_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """
    ModelTrainer class is used to train the model.
    """

    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Initiating model training")
            logging.info("Splitting the data into train and test")

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost Classifier": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostingRegressor": CatBoostRegressor(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
            }

            # TODO: Hyperparameter tuning

            model_report: dict = evaluate_models(models=models, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

            sorted_report = sorted(model_report.items(), key=lambda x: x[1]['r2_score'], reverse=True)

            best_model_name, best_model_score = sorted_report[0]
            best_model_score = best_model_score['r2_score']

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            save_obj(
                file_path=self.config.trained_model_path,
                obj=best_model,
            )

            logging.info(f"Model saved")

            predicted = best_model.predict(x_test)
            score = r2_score(y_test, predicted)
            return score

        except Exception as e:
            logging.error(f"Error in initiate_model_training: {e}")
            raise CustomException(e, "Error in initiate_model_training")
