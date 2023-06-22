import pandas as pd
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap

from sqlalchemy import create_engine
from azure.identity import DefaultAzureCredential
from azure.appconfiguration import AzureAppConfigurationClient
from forms import AddressForm

import requests
import os

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
    form = AddressForm()
    suggestions = []  # Initialize the suggestions variable as an empty list
    coordinates = []

    if form.validate_on_submit():
        address = form.address.data

        # Bing API request for auto-completion
        suggestion_url = 'http://dev.virtualearth.net/REST/v1/Autosuggest'
        suggestion_params = {
            'query': address,
            'key': 'AqCmXCSXpin76Mn5hLW5CVVvqbY88Ae9TbqET5mzzwohbCvs-ZbboA-OWuAoNThc'
        }
        suggestion_response = requests.get(suggestion_url, params=suggestion_params)
        suggestions = suggestion_response.json().get('resourceSets')[0].get('resources')

        # Bing API request for geocoding
        geocode_url = 'http://dev.virtualearth.net/REST/v1/Locations'
        geocode_params = {
            'q': address,
            'key': 'AqCmXCSXpin76Mn5hLW5CVVvqbY88Ae9TbqET5mzzwohbCvs-ZbboA-OWuAoNThc'
        }
        geocode_response = requests.get(geocode_url, params=geocode_params)
        coordinates = geocode_response.json().get('resourceSets')[0].get('resources')[0].get('point').get('coordinates')

        return render_template('model.html', form=form, suggestions=suggestions, coordinates=coordinates)

    return render_template('model.html', form=form, suggestions=suggestions, coordinates=coordinates)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

