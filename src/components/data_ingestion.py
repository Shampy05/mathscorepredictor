import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    """
    This class is used to store the configuration of the data ingestion component.

    Attributes:
        train_data_path: str
            The path to the training data
        test_data_path: str
            The path to the test data
        raw_data_path: str
            The path to the raw data
    """
    train_data_path: str=os.path.join("artifacts", "train.csv")
    test_data_path: str=os.path.join("artifacts", "test.csv")
    raw_data_path: str=os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        This function is used to initiate the data ingestion process.
        """
        try:
            logging.info("Initiating data ingestion process")
            df = pd.read_csv('/Users/zeno/Desktop/MLProject/notebook/data/stud.csv')
            logging.info("Read the data as a dataframe")
            os.makedirs(os.path.dirname(self.config.train_data_path) , exist_ok=True)

            df.to_csv(self.config.raw_data_path, index=False, header=True)

            logging.info("Train and test data split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logging.info("Data ingestion process completed")

            return (
                self.config.train_data_path,
                self.config.test_data_path,
            )
        except Exception as e:
            logging.info("Error occurred")
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()