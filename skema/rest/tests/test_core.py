import pytest
from fastapi.testclient import TestClient
from skema.rest.api import app 

@pytest.fixture
def client():
    return TestClient(app)

def test_version_endpoint(client):
    """Unit test for version endpoint"""
    response = client.get("/version")
    assert response.status_code == 200

def test_healthcheck_endpoint(client):
    """Unit test for healthcheck endpoint"""
    response = client.get("/healthcheck")
    assert response.status_code in [200, 500]

def test_environment_variables_endpoint(client):
    """Unit test for environment-variables endpoint"""
    response = client.get("/environment-variables")
    assert response.status_code == 200
    
    env_vars = response.json()
    assert "SKEMA_GRAPH_DB_PROTO" in env_vars
    assert "SKEMA_GRAPH_DB_HOST" in env_vars
    assert "SKEMA_GRAPH_DB_PORT" in env_vars
    assert "SKEMA_RS_ADDRESS" in env_vars

    assert "SKEMA_MATHJAX_PROTOCOL" in env_vars
    assert "SKEMA_MATHJAX_HOST" in env_vars
    assert "SKEMA_MATHJAX_PORT" in env_vars
    assert "SKEMA_MATHJAX_ADDRESS" in env_vars

    assert "MIT_TR_ADDRESS" in env_vars
    assert "SKEMA_TR_ADDRESS" in env_vars
    assert "COSMOS_ADDRESS" in env_vars
    

