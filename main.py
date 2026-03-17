import threading, json, uvicorn
from pathlib import Path
from simulator import sensor_feed
from simulator.camera_feed import get_current_frame

DECISION_FILE = Path(__file__).parent / "last_decision.json"
CURRENT_IMAGE_FILE = Path(__file__).parent / "current_image.txt"


def run_agent():
    from agent.sage_agent import run_cycle
    import time
    print("SAGE agent started")

    while True:
        # write "thinking" status so dashboard shows progress
        DECISION_FILE.write_text(json.dumps({
            "assessment": "Agent is thinking...",
            "reasoning": "",
            "actions_taken": [],
            "status": "running",
        }))

        # update current image path for dashboard
        data = sensor_feed.get()
        img = get_current_frame(data.get("scenario", "normal"))
        if img:
            CURRENT_IMAGE_FILE.write_text(img)

        def on_status(msg):
            DECISION_FILE.write_text(json.dumps({
                "assessment": msg,
                "reasoning": "",
                "actions_taken": [],
                "status": "running",
            }))

        result = run_cycle(status_callback=on_status)
        result["status"] = "done"
        DECISION_FILE.write_text(json.dumps(result))
        print(json.dumps(result, indent=2))
        time.sleep(30)


if __name__ == "__main__":
    sensor_feed.start()
    threading.Thread(target=run_agent, daemon=True).start()
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=8080, reload=False)