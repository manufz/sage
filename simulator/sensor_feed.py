import random, time, json, threading

SCENARIOS = {
    "normal":     {"moisture": 65, "temp": 22, "humidity": 70, "light": 800},
    "drought":    {"moisture": 18, "temp": 28, "humidity": 40, "light": 900},
    "overwater":  {"moisture": 92, "temp": 21, "humidity": 85, "light": 600},
    "heatstress": {"moisture": 50, "temp": 36, "humidity": 35, "light": 950},
}

current_scenario = "normal"
current_reading  = {}
_lock = threading.Lock()

def _loop():
    global current_reading
    while True:
        base = SCENARIOS[current_scenario]
        with _lock:
            current_reading = {
                k: round(v + random.uniform(-3, 3), 1)
                for k, v in base.items()
            } | {"ts": time.time(), "scenario": current_scenario}
        time.sleep(5)

def start():
    t = threading.Thread(target=_loop, daemon=True)
    t.start()

def get():
    with _lock:
        return dict(current_reading)

def set_scenario(name):
    global current_scenario
    if name in SCENARIOS:
        current_scenario = name