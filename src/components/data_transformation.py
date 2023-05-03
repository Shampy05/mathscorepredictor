import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocess_data_path: str = os.path.join("artifacts", "preprocess_data.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def initiate_data_transformation(self, train_data_path, test_data_path):
        """
        This function is used to initiate the data transformation process.
        :param train_data_path: path to the training data
        :param test_data_path: path to the test data
        :return: path to the preprocessed data
        """
        try:
            logging.info("Initiating data transformation process")

            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            logging.info("Read the data as a dataframe")

            preprocess_data = self.get_data_transformer_object()
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_data.drop([target_column_name], axis=1)
            target_feature_train_df = train_data[target_column_name]

            input_feature_test_df = test_data.drop([target_column_name], axis=1)
            target_feature_test_df = test_data[target_column_name]

            logging.info("Transforming the training data")

            input_feature_train_arr = preprocess_data.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocess_data.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving the preprocessed data")

            save_obj(
                file_path=self.config.preprocess_data_path,
                obj=preprocess_data
            )

            return (
                train_arr,
                test_arr,
                self.config.preprocess_data_path,
            )
        except Exception as e:
            logging.info("Error occurred")
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        """
        This function is used to get the data transformer object which is used to transform the data.
        :return: data transformer object
        """
        try:
            numerical_features = ["writing_score", "reading_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("std_scaler", StandardScaler(with_mean=False)),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info("Categorical and numerical pipeline created")

            preprocess_pipeline = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features),
                ]
            )

            logging.info("Preprocess pipeline created")

            return preprocess_pipeline
        except Exception as e:
            return CustomException(e, sys)