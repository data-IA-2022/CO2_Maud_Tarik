import pandas as pd
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap

from sqlalchemy import create_engine
from azure.identity import DefaultAzureCredential
from azure.appconfiguration import AzureAppConfigurationClient

app = Flask(__name__)



bootstrap = Bootstrap(app)

@app.route('/')
def index():
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

    return render_template('index.html', table_html=table_html, dept_number='')


# page analyse 

@app.route('/analyse')
def analyse():
    
    return render_template('analyse.html')

@app.route('/model', methods=['GET', 'POST'])
def model():
    
    return render_template('model.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

