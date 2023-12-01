import pytest
from app import app
import json

api_url = "http://127.0.0.1:8000"

@pytest.fixture
def client():
    test_app = app.test_client()
    with test_app as client:
        yield client

#TEST 1
def test_client_list(client):

    response = client.get(api_url + '/client_list')
    client_list = json.loads(response.data)

    # TEST 1.1: Check if the status code is 200 (OK)
    assert response.status_code == 200

    # TEST 1.2: Check if the response is a list
    assert isinstance(client_list, list)

    # TEST 1.3: Check if the first three SK_ID_CURR values are as expected
    expected_sk_id_curr = [100001, 100005, 100013]
    for i in range(3):
        assert client_list[i] == expected_sk_id_curr[i]

#TEST 2
def test_client_data(client):
    
    # Set up the SK_ID_CURR to the desired value
    sk_id_curr = 100001
    response = client.get(api_url + '/client_data', query_string={'SK_ID_CURR': str(sk_id_curr)})
    client_data = json.loads(response.data)

    # TEST 2.1: Check if the status code is 200 (OK)
    assert response.status_code == 200

    # TEST 2.2: Check if the number of features returned is 564
    assert len(client_data) == 564

    # TEST 2.3: Check if the values for columns for SK_ID_CURR = 100001 are as expected
    expected_values = {
        'CODE_GENDER': '1',
        'AMT_CREDIT': '568800.0'}

    for column, expected_value in expected_values.items():
        # Get the column value from the nested structure
        column_value = client_data.get(column)

        # If the column value is a dictionary, check the 'values' key
        actual_value = column_value.get('values') if isinstance(column_value, dict) else str(column_value)

        # Check if the actual value matches the expected value
        assert actual_value == expected_value
            
#TEST 3
def test_predict_default(client):

    # Set up the SK_ID_CURR to the desired value
    sk_id_curr = 100013
    response = client.get(api_url + '/predict_default', query_string={'SK_ID_CURR': str(sk_id_curr)})
    result = json.loads(response.data)

    # TEST 3.1: Check if the status code is 200 (OK)
    assert response.status_code == 200

    # TEST 3.2: Check if the predicted default probability for SK_ID_CURR = 100001 is as expected
    assert result["SK_ID_CURR"] == sk_id_curr
    assert result["default_probability"] >= 0 and result["default_probability"] <= 0.45