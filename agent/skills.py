import json, time
from pathlib import Path
from openai import OpenAI
import base64

import sys
sys.path.append(str(Path(__file__).parent.parent))
from simulator import sensor_feed, camera_feed

CROPS_PATH = Path(__file__).parent.parent / "crops.json"
ACTIONS_LOG = Path(__file__).parent.parent / "actions.log"

vision_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# --- Tool 1 ---
def read_sensors() -> dict:
    """Return current greenhouse sensor readings."""
    return sensor_feed.get()

# --- Tool 2 ---
def analyze_plant() -> str:
    """Run vision model on the current camera frame, return health assessment."""
    data = sensor_feed.get()
    frame_path = camera_feed.get_current_frame(data.get("scenario", "normal"))
    if not frame_path:
        return "No image available"

    with open(frame_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    response = vision_client.chat.completions.create(
        model="llava",  # swap for NVIDIA vision model if available on GB10
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text",
                 "text": (
                    "You are a plant health expert. In 1-2 sentences, assess this plant: "
                    "is it healthy, drought-stressed, overwatered, or showing disease? "
                    "Be specific about visible symptoms."
                 )}
            ]
        }],
        max_tokens=120,
    )
    return response.choices[0].message.content.strip()

# --- Tool 3 ---
def lookup_crop_needs(crop: str = "tomato") -> dict:
    """Return optimal growing conditions for the given crop."""
    crops = json.loads(CROPS_PATH.read_text())
    return crops.get(crop.lower(), crops["tomato"])

# --- Tool 4 ---
def trigger_action(action_type: str, value: str, detail: str = "") -> str:
    """Log an actuator command. action_type: irrigation|lights|fan|alert"""
    entry = {
        "ts": time.time(),
        "type": action_type,
        "value": value,
        "detail": detail,
    }
    with open(ACTIONS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return f"Action logged: {action_type} → {value}"