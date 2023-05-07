import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_obj(file_path, obj):
    """
    This function is used to save the preprocessed data.
    :param file_path: path to the preprocessed data
    :param obj: preprocessed data
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            dill.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(models, x_train, y_train, x_test, y_test):
    """
    This function is used to evaluate the models.
    :param models: models to be evaluated
    :param x_train: training data
    :param y_train: training data labels
    :param x_test: testing data
    :param y_test: testing data labels
    :return: model report
    """
    try:
        model_report = {}
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            model_report[model_name] = {
                "r2_score": r2_score(y_test, y_pred),
            }
        return model_report
    except Exception as e:
        raise CustomException(e, sys)
