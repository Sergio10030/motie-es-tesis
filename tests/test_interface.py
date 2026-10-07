"""Pruebas unitarias para la interfaz web."""
import pytest

from src.interface.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_index(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_reportante_get(client):
    resp = client.get("/reportante")
    assert resp.status_code == 200


def test_reportante_post(client):
    resp = client.post("/reportante", json={"texto": "estudiante golpeó a compañero"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert "id" in data


def test_reportante_post_empty(client):
    resp = client.post("/reportante", json={"texto": ""})
    assert resp.status_code == 400


def test_decisor_get(client):
    resp = client.get("/decisor")
    assert resp.status_code == 200


def test_evaluador_get(client):
    resp = client.get("/evaluador")
    assert resp.status_code == 200


def test_api_incidents(client):
    client.post("/reportante", json={"texto": "prueba de incidencia"})
    resp = client.get("/api/incidents")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"