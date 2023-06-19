import pandas as pd
import numpy as np

from utils import db_azure_connect, get_sod_data, haversine_distance, calculate_angle
from sqlalchemy import create_engine, types, text
from sqlalchemy.engine.reflection import Inspector

from sqlalchemy import create_engine, types
from azure.identity import DefaultAzureCredential
from azure.appconfiguration import AzureAppConfigurationClient


credential = DefaultAzureCredential()
appconfig_conn_str = "Endpoint=https://app-co2-config.azconfig.io;Id=8/Iv;Secret=8qfVLXI2aDt1Wg0jPMnCLx5lHIDvdAuzucBti8of7+M="
appconfig_client = AzureAppConfigurationClient.from_connection_string(connection_string=appconfig_conn_str)

# Retrieve the connection string from Azure App Configuration
setting = appconfig_client.get_configuration_setting(key="pg-connect-string")
secret_value = setting.value

# Create the SQLAlchemy engine
engine = create_engine(secret_value)
# Create the SQLAlchemy inspector
inspector = Inspector.from_engine(engine)

# drop existing table
with engine.begin() as conn:
        print(conn)
        conn.execute(text(f"""DROP TABLE IF EXISTS super_table_tm;  """))

## 1. Import du dataset connection api seattle data

data_ids_dict = {'2016' : '2bpz-gwpy', '2017' : 'qxjw-iwsh', '2018' : 'ypch-zswb', '2019' : '3th6-ticf', '2020' : 'auez-gz8p', '2021' : 'bfsh-nrm6'}

df_list_per_year = []

# Define the data types for each column
data_types = {
    'id' :  types.Integer(),
    'osebuildingid': types.Integer().with_variant(types.Integer, "postgresql"),
    'datayear': types.SmallInteger,
    'buildingtype': types.String(length=20),
    'primarypropertytype': types.String(length=70),
    'propertyname': types.String(length=72),
    'address': types.String(length=41),
    'city': types.String(length=8),
    'state': types.String(length=2),
    'zipcode': types.String(length=5),
    'taxparcelidentificationnumber': types.String(length=25),
    'councildistrictcode': types.SmallInteger,
    'neighborhood': types.String(length=53),
    'latitude': types.Float,
    'longitude': types.Float,
    'yearbuilt': types.SmallInteger,
    'numberofbuildings': types.SmallInteger,
    'numberoffloors': types.SmallInteger,
    'propertygfatotal': types.Integer,
    'propertygfaparking': types.Integer,
    'propertygfabuildings': types.Float,
    'listofallpropertyusetypes': types.String(length=255),
    'largestpropertyusetype': types.String(length=52),
    'largestpropertyusetypegfa': types.Float,
    'energystarscore': types.Float,
    'siteeuikbtusf': types.Float,
    'siteeuiwnkbtusf': types.Float,
    'sourceeuikbtusf': types.Float,
    'sourceeuiwnkbtusf': types.Float,
    'siteenergyusekbtu': types.Float,
    'siteenergyusewnkbtu': types.Float,
    'steamuse_kbtu': types.Float,
    'electricitykwh': types.Float,
    'electricitykbtu': types.Float,
    'naturalgastherms': types.Float,
    'naturalgaskbtu': types.Float,
    'defaultdata': types.String(length=5),
    'compliancestatus': types.String(length=28),
    'totalghgemissions': types.Float,
    'ghgemissionsintensity': types.Float,
    'secondlargestpropertyusetype': types.String(length=52),
    'secondlargestpropertyusetypegfa': types.Float,
    'thirdlargestpropertyusetype': types.String(length=52),
    'thirdlargestpropertyusetypegfa': types.Float,
    'yearsenergystarcertified': types.String(length=60),
    'log_totalghgemissions': types.Float,
    'log_siteenergyusekwh': types.Float,
    'is_using_steamusekWh': types.Float,
    'is_using_electricitykWh': types.Float,
    'is_using_naturalgaskWh': types.Float,
    'haversinedistance': types.Float,
    'angle': types.Float,
}



for data_year, data_id in data_ids_dict.items():
    data = get_sod_data(10000, dataset_ids= data_id)  # Retrieve the first 10000 records from the default dataset
    print(f"-------------------------------------------{data_year}----------------------------------------------")
    # Apply strip space and lowercase transformations to column names
    data = data.rename(columns=lambda x: x.strip().replace('_', '').replace('buildingname', 'propertyname'))
    
    
    # correction number of floor chinese baptist church osebuildingid: 21611
    data.loc[data['osebuildingid'] == '21611', 'numberoffloors'] = '1'
    
    if data_year in'2018':
        data.drop(['epabuildingsubtypename', 'complianceissue'], axis=1, inplace=True)
    if data_year in'2019':
        data.drop(['complianceissue', 'epapropertytype'], axis=1, inplace=True)
    if data_year in'2020':
        data.drop(['complianceissue', 'epapropertytype'], axis=1, inplace=True)
    if data_year in'2021':
        data.drop(['complianceissue', 'epapropertytype'], axis=1, inplace=True)

            
    if 'yearsenergystarcertified' in data.columns and 'outlier' in data.columns:
        data.drop(['yearsenergystarcertified', 'outlier'], axis=1, inplace=True)
    if '2016' in data_year or '2017' in data_year or '2019' in data_year or '2020' in data_year or '2021' in data_year:
        column_mapping = {
            'secondlargestpropertyuse': 'secondlargestpropertyusetypegfa'
        }

        data = data.rename(columns=column_mapping)
    
    if '2018' in data_year:
        column_mapping = {
            'largestpropertyusetype1': 'largestpropertyusetypegfa',
            'secondlargestpropertyuse1': 'secondlargestpropertyusetypegfa',
            'secondlargestpropertyuse': 'secondlargestpropertyusetype',
            'thirdlargestpropertyuse1': 'thirdlargestpropertyusetypegfa',
            'thirdlargestpropertyuse': 'thirdlargestpropertyusetype'
        }

        data = data.rename(columns=column_mapping)
    
    # Add datayear to osebuildingid as primary key
    data['id'] = data.apply(lambda row: f"{row['osebuildingid']}{row['datayear']}", axis=1)
    # Move the 'id' column to index 0
    data = data.reindex(columns=['id'] + data.columns[:-1].tolist())
    # Filter the DataFrame to keep only data with Compliant in ComplianceStatus
    data = data[data["compliancestatus"] == 'Compliant']
    # Drop the column after check only compliance in compliancesstatus
    data.drop(['compliancestatus', 'electricitykwh'], axis=1, inplace=True)
    # Filter the DataFrame to keep only rows where siteenergyusekbtu is not null
    data = data[data["siteenergyusekbtu"].notnull()]
    # fill Nan Null with np.nan
    data = data.fillna(np.nan)
    # Replace "NULL" with np.nan in your data
    data = data.replace("NULL", np.nan).replace("NA", np.nan)
    # # Replace "NULL" with np.nan in your data
    # data = data.replace("NA", np.nan)
    # Replace the 'City' column with the most frequent value
    most_frequent_city = data['city'].mode().iloc[0]
    data['city'] = most_frequent_city
    # Replace the 'City' column with the most frequent value
    most_frequent_city = data['state'].mode().iloc[0]
    data['state'] = most_frequent_city
    # check data type  
    # print(data["electricitykbtu"].dtype)
    # Convert kBtu to kWh in the Electricity column
    data["electricitykWh"] = pd.to_numeric(data["electricitykbtu"], errors="coerce") * 0.29307107

    # Convert kBtu to kWh in the SteamUse column
    data['steamusekWh'] = pd.to_numeric(data['steamusekbtu'], errors="coerce") * 0.29307107

    # Convert kBtu to kWh in the NaturalGas column
    data['naturalgaskWh'] = pd.to_numeric(data['naturalgaskbtu'], errors="coerce") * 0.29307107

    # Convert kBtu to kWh in the SiteEnergyUse column
    data['siteenergyusekWh'] = pd.to_numeric(data['siteenergyusekbtu'], errors="coerce") * 0.29307107

    # Convert square feet to square meters
    data['propertygfabuildings'] = pd.to_numeric(data['propertygfabuildings'], errors="coerce")
    data['propertygfabuildingm2'] = data['propertygfabuildings'] * 0.092903

    # Apply logarithmic transformation to columns with offset
    offset = 1  # Define the offset value
    data[['log_totalghgemissions', 'log_siteenergyusekwh']] = np.log(data[['totalghgemissions', 'siteenergyusekWh']].astype(float) + offset)
    cols = ['steamusekWh', 'electricitykWh', 'naturalgaskWh']
    for col in cols:
        #print(df[col].head())
        data['is_using_'+col] = data[col].apply(lambda x: 0 if x == 0 else 1)

    # Convert latitude and longitude columns to floating-point numbers
    data['latitude'] = data['latitude'].astype(float)
    data['longitude'] = data['longitude'].astype(float)
    
    # Center of Seattle coordinates
    seattle_center_lat = 47.6062
    seattle_center_lon = -122.3321

    # Calculate Haversine distance and angle for each row in the DataFrame
    data['haversinedistance'] = data.apply(lambda row: haversine_distance(
        seattle_center_lat, seattle_center_lon, row['latitude'], row['longitude']), axis=1)

    data['angle'] = data.apply(lambda row: calculate_angle(
        seattle_center_lat, seattle_center_lon, row['latitude'], row['longitude']), axis=1)
    # drop duplicate in 2021 
    data.drop_duplicates(subset=['osebuildingid'], inplace=True)
    # Add data_year to the df name
    variable_name = f"data_{data_year}"
    globals()[variable_name] = data
    df_list_per_year.append(globals()[variable_name])
    print(data.columns)
            
    db_azure_connect(df = data, data_types=data_types, table_name = 'super_table_tm')
    # with engine.connect() as conn:
    #     data_year = 'your_data_year'  # Replace with the desired data year
    #     conn.execute(text(f"""DELETE FROM "super_table_tm" WHERE "datayear" = '{data_year}'"""))
    #     print("Data deleted from super_table_tm table.")
    
    # db_azure_connect(df = data, data_types=data_types, table_name = 'super_table_tm')


# Pour une raison qui m'est inconnue le ty de certaines colonne ne change pas et rest en string
# Nous allons modidier la table après chargement des données 
# Execute the ALTER TABLE statements

# alter_statements = [
#     "ALTER TABLE public.super_table_tm ALTER COLUMN siteeuikbtusf TYPE float8 USING siteeuikbtusf::float8;",
#     "ALTER TABLE public.super_table_tm ALTER COLUMN siteeuiwnkbtusf TYPE float8 USING siteeuiwnkbtusf::float8;",
#     "ALTER TABLE public.super_table_tm ALTER COLUMN sourceeuikbtusf TYPE float8 USING sourceeuikbtusf::float8;",
#     "ALTER TABLE public.super_table_tm ALTER COLUMN sourceeuiwnkbtusf TYPE float8 USING sourceeuiwnkbtusf::float8;",
#     "ALTER TABLE public.super_table_tm ALTER COLUMN siteenergyusekbtu TYPE float8 USING siteenergyusekbtu::float8;",
#     "ALTER TABLE public.super_table_tm ALTER COLUMN siteenergyusewnkbtu TYPE float8 USING siteenergyusewnkbtu::float8;",
#     "ALTER TABLE public.super_table_tm ALTER COLUMN steamusekbtu TYPE float8 USING steamusekbtu::float8;",
#     "ALTER TABLE public.super_table_tm ALTER COLUMN electricitykbtu TYPE float8 USING electricitykbtu::float8;",
#     "ALTER TABLE public.super_table_tm ALTER COLUMN naturalgastherms TYPE float8 USING naturalgastherms::float8;",
#     "ALTER TABLE public.super_table_tm ALTER COLUMN naturalgaskbtu TYPE float8 USING naturalgaskbtu::float8;"
# ]


# with engine.connect() as conn:
#     print(conn)
#     for statement in alter_statements:
#         conn.execute(statement)
    
  
  

# print(len(df_list_per_year))
print()
# print(data["compliancestatus"].unique())
print(f"--------------------- Table totale --------------------------")
# # print(df_list_per_year )
# # Print the maximum length for each column
# # Iterate over the columns and calculate the maximum length
# max_len = {}
# for column in data.columns:
#     max_len[column] = data[column].astype(str).str.len().max()

# # Print the maximum length for each column
# for column, length in max_len.items():
#     print(f"{column}: {length}")

# for df in df_list_per_year:
#     # db_azure_connect(df = df, data_types=data_types, table_name = 'super_table_tm')
    
#     print(df.column())
# print(data_2018.head(3))


