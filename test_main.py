from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_chat_endpoint():
    response = client.post("/chat/", json={"question": "سؤال شما در اینجا"})
    assert response.status_code == 200
    assert isinstance(response.json(), dict)