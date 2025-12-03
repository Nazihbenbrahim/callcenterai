# tests/conftest.py
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

# 🔧 Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.tfidf_service.main import app as tfidf_app
from src.services.transformer_service.main import app as transformer_app
from src.services.agent_service.main import app as agent_app


@pytest.fixture(scope="session")
def tfidf_client():
    # ✅ startup/shutdown sont exécutés
    with TestClient(tfidf_app) as client:
        yield client


@pytest.fixture(scope="session")
def transformer_client():
    with TestClient(transformer_app) as client:
        yield client


@pytest.fixture(scope="session")
def agent_client():
    with TestClient(agent_app) as client:
        yield client
