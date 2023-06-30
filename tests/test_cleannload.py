import pytest
from cleanNload import *
from utils import *
from sqlalchemy import create_engine
import pandas as pd

@pytest.fixture(scope='module')
def db_test_engine():
    # Créer le moteur de base de données de test
    engine = create_engine(secret_value)


    # Nettoyer la base de données de test après les tests
    with engine.connect() as conn:
        conn.execute('DROP TABLE IF EXISTS test_table')

    yield engine

    engine.dispose()

def test_db_azure_connect(db_test_engine):
    # Créer la table de test dans la base de données
    with db_test_engine.connect() as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS test_table (id INT, name TEXT)')


    # Créer un DataFrame test
    test_data = pd.DataFrame({'id': [1, 2, 3], 'name': ['John', 'Jane', 'Joe']})

    # Appeler la fonction de connexion à la base de données
    db_azure_connect(df=test_data, data_types={}, table_name='test_table')

    # Vérifier si les données ont été insérées dans la base de données de test
    with db_test_engine.connect() as conn:
        result = conn.execute('SELECT COUNT(*) FROM test_table')
        count = result.scalar()
        assert count == len(test_data), "The number of inserted rows does not match the number of rows in the test DataFrame"

#test_db_azure_connect(db_test_engine)