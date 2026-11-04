"""Web layer: UI serving + upload validation. No LLM (mocked), no worker threads."""
import os
import tempfile

os.environ["NARRATE_MOCK"] = "1"  # must precede app imports
_td = tempfile.mkdtemp()
os.environ["INPUT_DIR"] = os.path.join(_td, "in")
os.environ["OUTPUT_DIR"] = os.path.join(_td, "out")

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)  # no `with`: lifespan threads stay off


def test_ui_served_without_tables():
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="drop"' in r.text
    assert "<table" not in r.text.lower()  # readability: card layout only


def test_pico_css_is_local():
    r = client.get("/static/pico.min.css")
    assert r.status_code == 200 and len(r.content) > 10_000

def test_upload_rejects_non_pdf():
    r = client.post("/api/upload", files={"files": ("notes.txt", b"hello")})
    assert r.status_code == 400
    assert "PDFs only" in r.json()["detail"]


def test_upload_pdf_registers_job():
    r = client.post("/api/upload", files={"files": ("mypaper.pdf", b"%PDF-1.4 fake")})
    assert r.status_code == 200
    assert r.json()["saved"] == ["mypaper.pdf"]
    names = [j["name"] for j in client.get("/api/jobs").json()]
    assert "mypaper" in names
