"""Shared test doubles and fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_extractor


class FakeExtractor:
    """Test double for the Extractor protocol: returns canned findings."""

    def __init__(self, findings):
        self._findings = findings

    async def extract(self, message: str):
        return self._findings


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def inject_findings():
    """Replace the extractor with canned findings.

    Usage: ``inject_findings([Finding(...)])`` inside a test.
    Overrides are cleaned up automatically.
    """
    def _inject(findings):
        app.dependency_overrides[get_extractor] = lambda: FakeExtractor(findings)

    yield _inject
    app.dependency_overrides.clear()
