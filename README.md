# SAGE — Sensor-Aware Greenhouse Engine

> Autonomous greenhouse AI agent built for the NVIDIA GTC 2026 Hack for Impact hackathon — Environmental Impact track.

SAGE monitors a tomato crop in real time using simulated sensors and a plant camera feed, reasons about plant health using a local vision model and LLM, and autonomously triggers corrective actions — entirely on-device with no cloud dependency.

---

## What it does

Every 30 seconds SAGE runs a full autonomous assessment cycle:

1. **Reads sensors** — soil moisture, temperature, humidity, light levels
2. **Analyzes the plant** — runs a vision model on the current camera frame to detect stress, disease, or anomalies
3. **Looks up crop requirements** — compares readings against optimal ranges for the current crop
4. **Decides and acts** — triggers irrigation, lighting, fan commands, or farmer alerts as needed
5. **Explains itself** — produces a plain-English reasoning trace visible on the live dashboard

---

## Demo

The live dashboard at `http://localhost:8080` shows:

- **Live sensor panel** — four sensor readings updating every 5 seconds
- **AI assessment panel** — current plant image, agent reasoning, and actions taken
- **Scenario trigger buttons** — instantly switch between Normal / Drought / Disease / Overwater / Heat stress
- **Sensor history charts** — moisture and temperature over time
- **Action log** — timestamped record of every AI decision

### Demo flow

1. Start on **Normal** — healthy crop, all sensors green
2. Click **Drought** — moisture drops to ~17%, temperature rises to ~28°C
3. Wait ~60 seconds — SAGE detects drought stress visually and via sensors, triggers irrigation
4. Click **Disease** — plant images switch to diseased leaves; SAGE detects through vision model even when sensors look normal
5. Watch the action log populate with SAGE's decisions and reasoning

---

## Architecture

```
Sense  →  Think  →  Act
```

```
simulator/
  sensor_feed.py       # emits soil moisture, temp, humidity, light every 5s
  camera_feed.py       # rotates through PlantVillage images by scenario
  images/              # 9,000+ tomato images: healthy, drought, disease, overwater

agent/
  skills.py            # 4 tools: read_sensors, analyze_plant, lookup_crop_needs, trigger_action
  sage_agent.py        # LLM tool-calling loop — sense, reason, act

dashboard/
  app.py               # FastAPI backend — /api/state, /api/decision, /api/actions, /api/image
  static/index.html    # live dashboard UI

crops.json             # optimal growing ranges for tomato, lettuce, basil
main.py                # boots sensor simulator + agent loop + dashboard server
```

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Hardware | Lenovo ThinkStation PGX — GB10, 128GB unified memory |
| LLM | llama3.2 via Ollama (upgrade path: NVIDIA Nemotron via NIM) |
| Vision | llava via Ollama (upgrade path: NVIDIA Cosmos VLM via NIM) |
| Model server | Ollama on port 11434, OpenAI-compatible API |
| Agentic layer | OpenClaw installed via clawspark on GB10 |
| Backend | FastAPI + uvicorn |
| Frontend | Vanilla HTML/JS + Chart.js |
| Dataset | PlantVillage — 9,000+ labeled tomato plant images |
| Dev workflow | rsync + SSH tunneling to GB10 |

---

## Running SAGE

### Prerequisites

On the GB10:
```bash
# install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# start Ollama and pull models
ollama serve &
ollama pull llama3.2
ollama pull llava

# install Python dependencies
pip install openai fastapi uvicorn pillow --break-system-packages
```

### Deploy from your Mac

```bash
# sync code to GB10 and run
rsync -av --exclude '__pycache__' --exclude '.git' --exclude '.DS_Store' \
  /Users/manu/SAGE/ nvidia@LENOVO-11.local:~/greenhouse-agent/

ssh nvidia@LENOVO-11.local "cd ~/greenhouse-agent && python3 main.py"
```

### Open the dashboard

In a separate terminal, set up the SSH tunnel:
```bash
ssh -L 8080:localhost:8080 nvidia@LENOVO-11.local -N
```

Then open `http://localhost:8080` in your browser.

### One-command deploy alias

Add to your Mac's `~/.zshrc`:
```bash
alias sage-deploy="rsync -av --exclude '__pycache__' --exclude '.git' --exclude '.DS_Store' \
  /Users/manu/SAGE/ nvidia@LENOVO-11.local:~/greenhouse-agent/ && \
  ssh nvidia@LENOVO-11.local 'cd ~/greenhouse-agent && python3 main.py'"
```

---

## Upgrade path to full NVIDIA stack

SAGE is built on the OpenAI-compatible API pattern throughout. Switching from Ollama to NVIDIA NIM is a one-line change in `sage_agent.py` and `skills.py`:

```python
# current (Ollama)
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# upgrade (NVIDIA NIM)
client = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed")
```

Model name swaps:
- `llama3.2` → `nvidia/nemotron`
- `llava` → `nvidia/cosmos-reason-vl`

---

## Dataset

SAGE uses the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) — 54,000+ labeled plant images across 38 disease classes. For SAGE we use a subset of tomato images organized into four scenario folders:

```
images/
  healthy/    # 1,591 images
  disease/    # 4,079 images (early blight, bacterial spot, leaf mold)
  drought/    # 1,591 images
  overwater/  # 1,591 images
```

---

## Environmental impact

Precision irrigation — SAGE only waters when actual plant stress is detected, not on a fixed timer. In real deployments this translates to 20-50% water savings. Running fully locally on GB10 hardware means SAGE is viable for remote farms without reliable internet connectivity — exactly where food security challenges are most acute.

---

## Hackathon

- **Event:** NVIDIA GTC 2026 — Hack for Impact: The Open Source AI Challenge
- **Track:** Environmental Impact
- **Hardware:** Lenovo ThinkStation PGX (GB10 / DGX Spark)
- **Team:** Manuel
