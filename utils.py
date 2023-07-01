import pandas as pd
import numpy as np
import math

from sqlalchemy import create_engine, types
from azure.identity import DefaultAzureCredential
from azure.appconfiguration import AzureAppConfigurationClient

import pandas as pd
from sodapy import Socrata




def get_sod_data(limit: int, dataset_ids: str = "2bpz-gwpy") -> pd.DataFrame:
  """
  Retrieves data from the specified dataset using the Socrata API.

  Args:
      limit (int): The maximum number of records to retrieve.
      dataset_ids (str): The ID of the dataset to retrieve from (default is "2bpz-gwpy").

  Returns:
      pd.DataFrame: A pandas DataFrame containing the retrieved data.
  """

  # Unauthenticated client only works with public data sets. Note 'None'
  client = Socrata("data.seattle.gov", None)

  # Get the dataset
  results = client.get(dataset_ids, limit=limit)

  # Display all columns of the DataFrame
  pd.set_option('display.max_columns', None)
  pd.set_option('display.max_rows', None)

  # Convert to pandas DataFrame
  df = pd.DataFrame.from_records(results)
  return df


def compare_colums(df1,df2):
  """
    Compares the columns of two DataFrames and returns the differing columns.

    Args:
        df1 (DataFrame): The first DataFrame for column comparison.
        df2 (DataFrame): The second DataFrame for column comparison.

    Returns:
        tuple: A tuple containing two lists: 
            - The first list contains columns that exist in df1 but not in df2.
            - The second list contains columns that exist in df2 but not in df1.

    Notes:
        - The function assumes that both df1 and df2 are valid DataFrames with columns.
        - The function compares the column names of df1 and df2 and identifies the differing columns.
        - The differing columns are returned as separate lists in a tuple.
    """
  columns_1 = list(df1.columns) 
  columns_2 = list(df2.columns)
  same_columns=[]
  diff_columns_2=[]
  diff_columns_1=[]

  for col in columns_2:
      if col in columns_1:
          same_columns.append(col)
      else:
          diff_columns_2.append(col)
  for col in columns_1:
      if col not in columns_2:
          diff_columns_1.append(col)
  return diff_columns_1, diff_columns_2



# function to connect to datase on azure with app 

def db_azure_connect(df, data_types, table_name):
  """
    Connects to an Azure SQL Database using SQLAlchemy and inserts a DataFrame into a specified table.

    Args:
        df (DataFrame): The DataFrame to be inserted into the database.
        data_types (dict): A dictionary specifying the data types of the DataFrame columns for proper SQL insertion.
        table_name (str, optional): The name of the table to insert the DataFrame into. Defaults to 'sod_origin_tm'.

    Returns:
        None

    Notes:
        - The function assumes that the necessary dependencies are imported and the required credentials and connection strings are correctly set up.
        - The function uses AzureAppConfigurationClient to retrieve the database connection string from Azure App Configuration.
        - The DataFrame is converted to SQL and sent to the database using SQLAlchemy's to_sql() method.
        - The if_exists parameter in to_sql() is set to 'replace' to replace the table if it already exists.
        - After successful insertion, the function prints a success message and closes the database connection.
    """
  # Create the AzureAppConfiguration client
  credential = DefaultAzureCredential()
  appconfig_conn_str = "Endpoint=https://app-co2-config.azconfig.io;Id=8/Iv;Secret=8qfVLXI2aDt1Wg0jPMnCLx5lHIDvdAuzucBti8of7+M="
  appconfig_client = AzureAppConfigurationClient.from_connection_string(connection_string=appconfig_conn_str)

  # Retrieve the connection string from Azure App Configuration
  setting = appconfig_client.get_configuration_setting(key="pg-connect-string")
  secret_value = setting.value

  # Create the SQLAlchemy engine
  engine = create_engine(secret_value)
  print(engine)

  # Specify the table name
  table_name = table_name

  # Convert the DataFrame to SQL and send it to the database
  df.to_sql(table_name, engine, if_exists='append', index=False, dtype=data_types)


  print("Data has been successfully appended to the table.")

  # Close the database connection
  engine.dispose()


# Function to calculate Haversine distance
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points
    on the Earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = 6371 * c  # Radius of the Earth in kilometers
    return distance

# Function to calculate angle
def calculate_angle(lat1, lon1, lat2, lon2):
    """
    Calculate the angle (bearing) between two points
    on the Earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Calculate the angle using trigonometry
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    angle = math.atan2(y, x)
    angle = math.degrees(angle)
    angle = (angle + 360) % 360  # Convert to positive angle
    return angle


data_ids_dict = {'2016' : '2bpz-gwpy', '2017' : 'qxjw-iwsh'}

df_list_per_year = []

# for data_year, data_id in data_ids_dict.items():
#   data = get_sod_data(3, dataset_ids= data_id)  # Retrieve the first 3 records from the default dataset
#   # Apply strip space and lowercase transformations to column names
#   data = data.rename(columns=lambda x: x.strip().replace('_', ''))
#   # Add datayear to osebuildingid as primary key
#   data['id'] = data.apply(lambda row: f"{row['osebuildingid']}{row['datayear']}", axis=1)
#   # Move the 'id' column to index 0
#   data = data.reindex(columns=['id'] + data.columns[:-1].tolist())
#   # Add data_year to the df name
#   variable_name = f"data_{data_year}"
#   globals()[variable_name] = data
#   df_list_per_year.append(globals()[variable_name])
  
#   # print(data.head())  
  
# print(len(df_list_per_year))
# # print(data_2017.head)

# for df_year in df_list_per_year[1:]:
#   diff_columns_ref, diff_columns_other_year = compare_colums(df_list_per_year[0],df_year)

#   print(diff_columns_ref,'\n','\n', diff_columns_other_year)

# # Perform column comparison between data frames
# for i in range(1, len(df_list_per_year)):
#     ref_df = df_list_per_year[0]
#     comp_df = df_list_per_year[i]
#     year = int(list(data_ids_dict.keys())[i])
#     diff_columns = ref_df.columns.difference(comp_df.columns)
#     variable_name = f"diff_columns_{year}"
#     exec(f"{variable_name} = diff_columns")

#     print(f"Differences between 2016 and {year}:")
#     if len(diff_columns) > 0:
#         print(diff_columns)
#     else:
#         print(f"No column differences found between 2016 and {year}")
#     print()

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer


def create_data_preparation(data):
    # ...

    # Variables catégorielles à transformer avec OneHotEncoder
    column_cat_onehot = ['buildingtype', 'primarypropertytype']
    transfo_cat_onehot = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # # Variables booléennes (sans traitement)
    # column_bool = ['is_using_steamusekWh', 'is_using_electricitykWh', 'is_using_naturalgaskWh']
    # transfo_bool = FunctionTransformer(validate=False)

    # Variables numériques
    column_numeric = ['yearbuilt', 'largestpropertyusetypegfa', 'numberofbuildings',
                      'numberoffloors', 'propertygfabuildings',
                      'is_using_steamusekWh', 'is_using_electricitykWh', 'is_using_naturalgaskWh']
    column_numeric = [col for col in column_numeric if col in data.columns]

    # Numeric data imputation with KNNImputer
    transfo_numeric = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5, weights='uniform')),
        ('scaling', RobustScaler())
    ])

    # Création du préparateur de données
    preparation = ColumnTransformer(transformers=[
        ('data_numeric', transfo_numeric, column_numeric),
        ('data_cat_onehot', transfo_cat_onehot, column_cat_onehot)#,
        # ('data_bool', transfo_bool, column_bool)
    ])

    return preparation

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, learning_curve
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
import joblib


def train_single_output_models(X, Y, preparation):
    
    """
    Train and evaluate single-output regression models using grid search.

    Args:
        X (numpy array or pandas DataFrame): Training feature data.
        Y (numpy array or pandas DataFrame): Training target data.
        preparation 

    Returns:
        models_compare_metrics (pandas DataFrame): DataFrame containing the model comparison metrics.
        learning_curves_data (list): List of tuples containing learning curve data for each model.
    Exemple of use:
    models_compare_metrics, learning_curves_data = train_single_output_models(X, Y, preparation)
    """
    
    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42) 
    #initialisation de 
    models_opti = []
    parameters = {}
    models_param = {
        # RandomForestRegressor: {
        #     'model__n_estimators': [100, 200, 500],
        #     'model__max_depth': [10]
        # },
        # xgb.XGBRegressor: {
        #     'model__n_estimators': [100, 200, 500],
        #     'model__max_depth': [10],
        #     'model__learning_rate': [0.1, 0.01, 0.001]
        # },
        
        GradientBoostingRegressor: {
            'model__loss': ['squared_error'],
            'model__n_estimators': [100, 200],
            'model__max_depth': [25],
            'model__learning_rate': [0.1, 0.01]
        }
    }

    model_names = [
        # 'RandomForestRegressor',
        # 'XGBRegressor',
        'GradientBoostingRegressor'
    ]

    metrics = ['train_time', 'test_time', 'R2_score_train', 'R2_score_test', 'MAE_train_score', 'MAE_test_score',
               'Best_parameters']
    models_compare_metrics = pd.DataFrame(columns=metrics)

    best_model = None
    best_test_score = -np.inf
    learning_curves_data = []

    for model, model_name in zip(models_param.keys(), model_names):
        pipeline = Pipeline(steps=[('preparation', preparation),('model', model())])

        parameters = models_param[model]

        gscv = GridSearchCV(pipeline, parameters, scoring='r2', cv=5, verbose=2)
        start_time = time.time()
        gscv.fit(X_train, Y_train)
        end_time = time.time()
        models_opti.append(gscv)

        elapsed_time = end_time - start_time
        print(model_name, gscv.best_score_, gscv.best_params_, "--- Time taken:", elapsed_time, "seconds")

        best_estimator = gscv.best_estimator_
        best_estimator.fit(X_train, Y_train)

        start_time = time.time()
        best_estimator.predict(X_train)
        train_time = time.time() - start_time

        start_time = time.time()
        best_estimator.predict(X_test)
        test_time = time.time() - start_time

        r2_score_train = best_estimator.score(X_train, Y_train)
        r2_score_test = best_estimator.score(X_test, Y_test)
        mae_train_score = mean_absolute_error(Y_train, best_estimator.predict(X_train))
        mae_test_score = mean_absolute_error(Y_test, best_estimator.predict(X_test))

        best_parameters = gscv.best_params_

        # Calculate the score on the test data
        test_score = best_estimator.score(X_test, Y_test)
        print(test_score)

        # Save the best model based on the test score
        if test_score > best_test_score:
            best_model = best_estimator
            best_test_score = test_score
            best_parameters = gscv.best_params_

        # Save the best model
        if best_model is not None:
            target_name = Y_train.columns[0]  # Assuming Y_train is a DataFrame with a single column
            joblib.dump(best_model, f'data/best_model_{model_name}_{target_name}.pkl')

        # Save the best parameters
        target_name = Y_train.columns[0]  # Assuming Y_train is a DataFrame with a single column
        joblib.dump(best_parameters, f'data/best_parameters_{model_name}_{target_name}.pkl')

        models_compare_metrics.loc[model_name] = [train_time, test_time, r2_score_train, r2_score_test,
                                                  mae_train_score, mae_test_score, best_parameters]

        print("R2 score train:", r2_score_train)
        print("R2 score test:", r2_score_test)
        print("MAE train score:", mae_train_score)
        print("MAE test score:", mae_test_score)
        print("Best parameters:", gscv.best_params_)

        # Generate learning curve data
        train_sizes, train_scores, test_scores = learning_curve(best_estimator, X_train, Y_train, cv=5, scoring='r2')
        learning_curves_data.append((model_name,target_name, train_sizes, train_scores, test_scores))

    # Display the comparison dataframe
    print(models_compare_metrics)

    return models_compare_metrics, learning_curves_data

import csv

def get_delimiter(file_path, bytes = 4096):
    sniffer = csv.Sniffer()
    data = open(file_path, "r").read(bytes)
    delimiter = sniffer.sniff(data).delimiter
    return delimiter


def process_csv(csv_file):
    delim = get_delimiter(csv_file, bytes = 4096)
    process_csv_file = pd.read_csv(csv_file, delimiter=delim)
    # fill Nan Null with np.nan
    process_csv_file = process_csv_file.fillna(np.nan)
    # Replace "NULL" with np.nan in your data
    process_csv_file = process_csv_file.replace("NULL", np.nan).replace("NA", np.nan)
    error = None
    df_pred = pd.DataFrame()
    if set(['BuildingType','PrimaryPropertyType','YearBuilt',
            'NumberofBuildings','NumberofFloors','PropertyGFABuilding(s)','LargestPropertyUseTypeGFA',
            'SteamUse(kBtu)', 'Electricity(kBtu)', 'NaturalGas(kBtu)']).issubset(process_csv_file.columns):
        
        
        # yearbuilt field
        df_pred['yearbuilt'] = process_csv_file['YearBuilt']
        # buildingtype field
        df_pred['buildingtype'] = process_csv_file['BuildingType']
        
        # primarypropertytype field
        df_pred['primarypropertytype'] = process_csv_file['PrimaryPropertyType']
        
        # primarypropertytype field
        df_pred['largestpropertyusetypegfa'] = process_csv_file['LargestPropertyUseTypeGFA'].astype(float).round(2)
        
        # propertygfabuildings field
        df_pred['propertygfabuildings'] = process_csv_file['PropertyGFABuilding(s)'].astype(float).round(2)
        
        # numberofbuildings field
        df_pred['numberofbuildings'] = process_csv_file['NumberofBuildings']
        
        # numberoffloors field
        df_pred['numberoffloors'] = process_csv_file['NumberofFloors']
        
        cols = ['SteamUse(kBtu)', 'Electricity(kBtu)', 'NaturalGas(kBtu)']
        predcols = ["is_using_steamusekWh", "is_using_electricitykWh", "is_using_naturalgaskWh"]
        for col,predcol in zip(cols, predcols):
            #print(df[col].head())
            df_pred[predcol] = process_csv_file[col].apply(lambda x: 0 if x == 0 else 1)

        # import model avec joblib
        loaded_model_energyuse = joblib.load(f'data/best_model_GradientBoostingRegressor_siteenergyusekbtu.pkl')
        loaded_model_ghgemissions = joblib.load(f'data/best_model_GradientBoostingRegressor_totalghgemissions.pkl')
        
        # prediction avec le modele
        y_pred_energyuse = loaded_model_energyuse.predict(df_pred)
        y_pred_ghgemissions = loaded_model_ghgemissions.predict(df_pred)
        ghgemissions = y_pred_ghgemissions
        energyuse = y_pred_energyuse
        process_csv_file['Predict_ghgemissions'] = ghgemissions
        process_csv_file['Predict_energyuse'] = energyuse
                
    return process_csv_file