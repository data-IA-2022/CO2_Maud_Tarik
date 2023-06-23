import pandas as pd
import numpy as np
import requests
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from azure.identity import DefaultAzureCredential
from azure.appconfiguration import AzureAppConfigurationClient
from forms import AddressForm
import json

from utils import haversine_distance

import os
from glob import glob

import joblib

app = Flask(__name__)

app.config['SECRET_KEY'] = os.urandom(24) # génération d'une clée pour le CSRF protection in Flask-WTF.

bootstrap = Bootstrap(app)

@app.route('/')
def index():
    suggestions = []
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
    
    
    # Query the entire table with LIMIT 10
    query = "SELECT * FROM co2_dataset LIMIT 10"

    df = pd.read_sql_query(query, engine)

    # Convert the DataFrame to an HTML table
    table_html = df.to_html(index=False)

    return render_template('index.html', table_html=table_html, dept_number='', suggestions=suggestions)


# page analyse 

@app.route('/analyse')
def analyse():
    suggestions = []
    return render_template('analyse.html', suggestions=suggestions)

@app.route('/model', methods=['GET', 'POST'])
def model():
    # Create the AzureAppConfiguration client
    credential = DefaultAzureCredential()
    appconfig_conn_str = "Endpoint=https://app-co2-config.azconfig.io;Id=8/Iv;Secret=8qfVLXI2aDt1Wg0jPMnCLx5lHIDvdAuzucBti8of7+M="
    appconfig_client = AzureAppConfigurationClient.from_connection_string(connection_string=appconfig_conn_str)
    
    # Retrieve the connection string from Azure App Configuration
    setting = appconfig_client.get_configuration_setting(key="pg-connect-string")
    secret_value = setting.value
    # Retrieve the Bing key API for geocoding from Azure App Configuration
    setting_bing = appconfig_client.get_configuration_setting(key="BING-API-KEY")
    bingApiKey = setting_bing.value
     
    # Create the SQLAlchemy engine
    engine = create_engine(secret_value)
    print(engine)
    
        
    suggestions = []
    coordinates = []
    latitude = None
    longitude = None
    
    # Get the model names from data file to populate the dropdown box
    model_files = glob('data/best_model_*.pkl')
    model_names = [os.path.basename(file)[:-4] for file in model_files]
    # Query the distinct building types
    query2 = "SELECT DISTINCT buildingtype FROM super_table_tm WHERE datayear = 2016"
    building_types = pd.read_sql_query(query2, engine)["buildingtype"].tolist()

    # Query the distinct primary property types
    query3 = "SELECT DISTINCT primarypropertytype FROM super_table_tm WHERE datayear = 2016"
    property_types = pd.read_sql_query(query3, engine)["primarypropertytype"].tolist()

    if request.method == 'POST':
        # adress field
        address = request.form['address']
        
        # yearbuilt field
        yearbuilt = request.form['yearbuilt']
        
        # buildingtype field
        buildingtype = request.form['buildingtype']
        
        # primarypropertytype field
        primarypropertytype = request.form['primarypropertytype']
        
        # primarypropertytype field
        largestpropertyusetypegfa = request.form['largestpropertyusetypegfa']
        
        # propertygfabuildings field
        propertygfabuildings = request.form['propertygfabuildings']
        
        # numberofbuildings field
        numberofbuildings = request.form['numberofbuildings']
        
        # numberoffloors field
        numberoffloors = request.form.get('numberoffloors')

        is_using_naturalgaskWh = request.form.get('is_using_naturalgaskWh', '0')
        
        is_using_steamusekWh = request.form.get('is_using_steamusekWh', '0')
        
        is_using_electricitykWh = request.form.get('is_using_electricitykWh', '0')
        
        # Bing API request for auto-completion
        suggestion_url = 'http://dev.virtualearth.net/REST/v1/Autosuggest'
        suggestion_params = {
            'query': address,
            'key': bingApiKey
        }
        suggestion_response = requests.get(suggestion_url, params=suggestion_params)
        suggestions = suggestion_response.json().get('resourceSets')[0].get('resources')

        # Bing API request for geocoding
        geocode_url = 'http://dev.virtualearth.net/REST/v1/Locations'
        geocode_params = {
            'q': address,
            'key': bingApiKey
        }
        geocode_response = requests.get(geocode_url, params=geocode_params)

        try:
            geocode_data = geocode_response.json()
            coordinates = geocode_data.get('resourceSets')[0].get('resources')[0].get('point').get('coordinates')
            latitude = coordinates[0]
            longitude = coordinates[1]
            # Center of Seattle coordinates
            seattle_center_lat = 47.6062
            seattle_center_lon = -122.3321

            # Calculate Haversine distance and angle for the request adress in the formular
            haversinedistance = haversine_distance(seattle_center_lat, seattle_center_lon, latitude, longitude)
            # Create the table of df_prediction
            df_input = pd.DataFrame({
                'haversinedistance': [haversinedistance],  # Wrap scalar value in a list
                'yearbuilt': [yearbuilt],
                'is_using_electricitykWh': [is_using_electricitykWh],
                'is_using_naturalgaskWh': [is_using_naturalgaskWh],
                'is_using_steamusekWh': [is_using_steamusekWh],
                'largestpropertyusetypegfa': [largestpropertyusetypegfa],
                'numberofbuildings': [numberofbuildings],
                'numberoffloors': [numberoffloors],
                'propertygfabuildings': [propertygfabuildings],
                'buildingtype': [buildingtype],
                'primarypropertytype': [primarypropertytype]
            }, index=[0])  # Add index sinon ValueError: If using all scalar values, you must pass an index
            
            # get selected model selected_model field
            selected_model = request.form['model']
            # import model avec joblib
            loaded_model = joblib.load(f'data/{selected_model}.pkl')
            # prediction avec le modele
            y_pred = loaded_model.predict(df_input)
            ghgemissions = np.exp(round(y_pred[0][0], 2))
            energyuse = np.exp(round(y_pred[0][1], 2))
            
        except json.decoder.JSONDecodeError as e:
            print('Error decoding JSON response:', e)
            print('Response content:', geocode_response.content)
        
                
        return render_template('model.html', building_types=building_types, property_types=property_types, suggestions=suggestions, coordinates=coordinates, latitude=latitude,
                               longitude=longitude, bing_maps_api_key=bingApiKey,table = df_input.to_html(),
                               model_names=model_names,prediction=y_pred.all(), ghgemissions=ghgemissions,energyuse=energyuse)
    return render_template('model.html', building_types=building_types, property_types=property_types, suggestions=suggestions, coordinates=coordinates, latitude=latitude,
                           longitude=longitude, bing_maps_api_key=bingApiKey,model_names=model_names)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

