from fastapi.testclient import TestClient

from skema.gromet.execution_engine.server import app

client = TestClient(app)

def test_enrich_amr():
    '''Test case for /amr-enrichment endpoint'''
    request = {
        "amr": {
            "semantics": {
                "ode":{
                    "parameters":[
                        {"name": "a"},
                        {"name": "b"},
                        {"name": "c"}
                    ]
                }
            }
        },
        "source": "a=1\nb=a+1\nc=b-a",
        "filename": "source.py"
    }

    response=client.post("/execution_engine/amr-enrichment", json=request)
    parameters = response.json()["semantics"]["ode"]["parameters"]
    assert parameters == [
        {"name": "a", "value": 1},
        {"name": "b", "value": 2},
        {"name": "c", "value": 1}
    ]
