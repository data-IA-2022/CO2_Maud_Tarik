import pandas as pd
import numpy as np

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



data_ids_dict = {'2016' : '2bpz-gwpy', '2017' : 'qxjw-iwsh', '2018' : 'ypch-zswb'}

df_list_per_year = []

for data_year, data_id in data_ids_dict.items():
  data = get_sod_data(3, dataset_ids= data_id)  # Retrieve the first 100 records from the default dataset
  # # Apply strip space and lowercase transformations to column names
  # data = data.rename(columns=lambda x: x.strip().lower())
  # # Add datayear to osebuildingid as primary key
  # data['id'] = data.apply(lambda row: f"{row['osebuildingid']}{row['datayear']}", axis=1)
  # # Move the 'id' column to index 0
  # data = data.reindex(columns=['id'] + data.columns[:-1].tolist())
  # Add data_year to the df name
  variable_name = f"data_{data_year}"
  globals()[variable_name] = data
  df_list_per_year.append(globals()[variable_name])
  
  # print(data.head())  
  
print(len(df_list_per_year))
# print(data_2017.head)

for df_year in df_list_per_year[1:]:
  diff_columns_ref, diff_columns_other_year = compare_colums(df_list_per_year[0],df_year)

  print(len(diff_columns_ref), len(diff_columns_other_year))

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