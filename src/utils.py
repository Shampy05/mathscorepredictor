import os
import sys

import numpy as np
import pandas as pd
import dill

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
