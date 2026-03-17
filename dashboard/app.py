from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from simulator import sensor_feed

app = FastAPI()

ACTIONS_LOG    = Path(__file__).parent.parent / "actions.log"
DECISION_FILE  = Path(__file__).parent.parent / "last_decision.json"
CURRENT_IMAGE  = Path(__file__).parent.parent / "current_image.txt"

@app.get("/api/state")
def get_state():
    return sensor_feed.get()

@app.get("/api/actions")
def get_actions():
    if not ACTIONS_LOG.exists():
        return []
    lines = ACTIONS_LOG.read_text().strip().split("\n")
    return [json.loads(l) for l in lines if l][-20:]

@app.get("/api/decision")
def get_decision():
    if not DECISION_FILE.exists():
        return {"assessment": "Waiting for first cycle...", "reasoning": "", "actions_taken": [], "status": "waiting"}
    return json.loads(DECISION_FILE.read_text())

@app.get("/api/scenario/{name}")
def set_scenario(name: str):
    sensor_feed.set_scenario(name)
    return {"ok": True, "scenario": name}

@app.get("/api/image")
def get_image():
    if not CURRENT_IMAGE.exists():
        return FileResponse("/dev/null", media_type="image/jpeg")
    path = CURRENT_IMAGE.read_text().strip()
    if path and Path(path).exists():
        return FileResponse(path, media_type="image/jpeg")
    return FileResponse("/dev/null", media_type="image/jpeg")

app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")