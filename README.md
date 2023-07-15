## Math Score Predictor

This project is an end-to-end data science application for predicting math scores of students based on various features. The project includes data ingestion, data transformation, model training, and a prediction pipeline. It aims to provide insights into how student performance is influenced by factors such as gender, ethnicity, parental level of education, lunch, and test preparation course.

## Project Structure

The project follows a modular structure with the following components:

- src/components: This directory contains all the modules created for the project.
  - data_ingestion.py: Module responsible for reading data from a database or file location. It also divides the data into training and testing sets.
  - data_transformation.py: Module for transforming the data, including handling categorical-to-numerical conversion, one-hot encoding, and label encoding.
  - model_trainer.py: Module for training different types of models and evaluating their performance, such as calculating the R2 value for regression models. It also performs data validation and evaluation.
  - (future) data_validation.py: Module for data validation and evaluation.
- src/pipeline: This directory contains the pipeline-related code.
  - train_pipeline.py: Code for training the pipeline. It triggers or calls the above components.
  - predict_pipeline.py: Code for making predictions using new data.
- src/logger.py: Module for logging the execution and tracking errors.
- src/exception.py: Module for creating custom detailed exceptions.
- src/utils.py: Utility module for various tasks such as reading data from a database.
  
## Project Lifecycle

The project follows the following lifecycle stages:

1. Understanding the problem: The project aims to understand how student performance (test score) is affected by variables such as gender, ethnicity, parental level of education, lunch, and test preparation course. The goal is to predict the test score based on these features.
2. Data collection: The dataset used for the project is obtained from Kaggle, consisting of 8 columns and 1000 rows.
3. Data checks:
  - Check missing values: Use df.isna().sum() to identify missing values. If missing values are found, mean imputation can be performed.
  - Check data types: Use df.info() to check the data types of columns.
  - Check duplicates: Use df.duplicated().sum() to identify duplicate rows.
  - Check number of unique values in each column: Use df.nunique() to count the unique values in each column.
  - Check statistics: Use df.describe() to obtain descriptive statistics of the dataset.
  - Check various categories in categorical columns: Identify categorical columns (numeric_feature = [feature for feature in df.columns if df[feature].dtype == 'o']) and examine the unique categories.
  - (future) Data collection from a database: In the future, the dataset could be obtained from a database.
4. Data preprocessing: Perform data preprocessing tasks such as adding columns for total_score and average_score. These columns will be the dependent features, while the rest are independent features.
5. Model training: Since the total score is a continuous value, this is a regression problem. Import different regressors (e.g., KNR, DecisionTreeRegressor, RandomForestRegressor, AdaBoostRegressor, LinearRegression, CatboostRegressor, XGBRegressor) and evaluate their performance to determine the best model.
6. Choosing the best model: Select the best-performing model based on evaluation metrics such as R2 score.
7. (future) Hyperparameter tuning: Perform hyperparameter tuning for each model by specifying parameters and applying GridSearchCV.

## Data Ingestion

1. Data Ingestion: In this step, data is collected from various sources and stored in a database or Hadoop. The data is then split, and data transformation takes place. For data ingestion, the inputs required by the data ingestion components (e.g., paths to save test, train, and raw data) should be provided. The DataIngestionConfig class is created using @dataclass to define the configuration parameters.
  - In DataIngestionConfig, the train_data_path class variable defines the path to store the training data (in the "artifacts" folder as "train.csv") and other data.
  - (future) Additional configuration: In the future, a separate folder such as "Config Entity" could be created for storing additional configuration files.
    
2. The DataIngestion class is defined, which initializes the DataIngestionConfig instance. It contains the initiate_data_ingestion function, which reads the dataset (using read_csv or other methods) and creates the necessary directories using os.makedirs based on the paths specified in DataIngestionConfig. The dataset is then split into training and testing sets, which are stored in their respective folders.
   
## Data Transformation
1. In the data transformation step, various transformations are applied to categorical and numerical features. The main objective is to perform feature engineering and data cleaning.
2. The following modules are imported from sklearn: ColumnTransformer, SimpleImputer, Pipeline, OneHotEncoder, and StandardScaler.
3. The DataTransformationConfig class is created to provide any necessary paths or inputs required for data transformation. For example, the path to save a model in a pickle file ("preprocessor.pkl") can be specified.
4. The DataTransformation class is created, which initializes the DataTransformationConfig instance. The get_data_transformer_obj function is defined to create the pickle files responsible for converting categorical features to numerical features. The function uses SimpleImputer with the median strategy and StandardScaler for the numerical pipeline. For the categorical pipeline, it uses SimpleImputer with the most frequent strategy, OneHotEncoder, and StandardScaler. The preprocessing steps are combined using the ColumnTransformer. The preprocessor is then returned.
5. The same class contains the initiate_data_transform function, which takes the paths to the training and testing data as inputs. Inside this function, the train_set, test_set, and preprocessor are initialized. The target column (math_score) is dropped from the train and test dataframes. The fit_transform method is then applied to both the train and test datasets, applying the scaling and transformation steps. Finally, the transformed datasets are horizontally concatenated using np.c_.
6. The save_obj function in utils.py is responsible for saving the pickle file based on the provided configuration location.

## Model Trainer

1. The model trainer step involves training various regression models.
2. The ModelTrainerConfig file is created to define the path for the model pickle file ("model.pkl").
3. The ModelTrainer class is defined, which trains the models and initializes the configuration using ModelTrainerConfig. The initiate_model_trainer function is created to take the training and testing arrays and the path to the preprocessor pickle file as inputs. The function splits the data into X_train, y_train, X_test, and y_test.
4. The following models are selected for training and evaluation: RandomForestRegressor, DecisionTreeRegressor, GradientBoostingRegressor, LinearRegression, KNeighboursRegressor, XGBRegressor, CatboostRegressor, and AdaBoostRegressor.
5. The model_report function defined in utils.py is used to iterate over each model, fit the model on X_train and y_train, and make predictions on X_train and X_test. The r2_score is used to evaluate the performance of each model.
6. The best model's score and name are extracted from the dictionary of model scores.
7. Hyperparameter tuning can be performed for each model by specifying parameters and applying GridSearchCV.

## Prediction Pipeline

The prediction pipeline involves creating a web application that interacts with the pickle files for making predictions.

- app.py is created as a Flask application, importing Flask, request, and render_pipeline. An entry point for Flask is defined.
- Routes are created for "/" and "/predictdata" (GET and POST). The POST request for "/predictdata" includes the CustomData class responsible for mapping the provided values from the frontend to the backend.
- 
Deployment using Elastic Beanstalk on AWS:

- Elastic Beanstalk (EB) is used to run and manage web applications in a cloud environment.
- EB requires configuration in a python.config file.
- Any changes made in the GitHub repository will be reflected and deployed in EB using Code Pipeline (Continuous Delivery Pipeline).

