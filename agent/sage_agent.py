import json, time, re
from openai import OpenAI
from agent.skills import read_sensors, analyze_plant, lookup_crop_needs, trigger_action

nim_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

SYSTEM_PROMPT = """
You are SAGE — an autonomous greenhouse AI managing a tomato crop.
Every cycle you will receive live sensor data and a plant health assessment.
You have four tools available:
  - read_sensors: get current soil moisture, temperature, humidity, light
  - analyze_plant: run vision model on the current camera frame
  - lookup_crop_needs: get optimal ranges for a crop
  - trigger_action: fire an actuator command (irrigation, lights, fan, alert)

Your job each cycle:
1. Call read_sensors and analyze_plant
2. Call lookup_crop_needs for the current crop
3. Reason step by step: compare readings to optimal ranges, consider the visual assessment
4. ALWAYS call trigger_action at least once — use these rules:
   - moisture below optimal minimum → trigger_action irrigation on
   - moisture above optimal maximum → trigger_action fan on
   - temperature above optimal maximum → trigger_action lights reduced
   - disease symptoms visible → trigger_action alert with treatment recommendation
   - all conditions normal → trigger_action alert ok
5. Return ONLY a valid JSON object with no markdown, no code fences, no extra text:
{"assessment": "one sentence plain-English plant status", "reasoning": "2-3 sentences explaining your decisions", "actions_taken": ["irrigation on 8min", "alert: drought stress detected"]}

RULES:
- You MUST call trigger_action every single cycle without exception
- Your final response must be pure JSON only — no markdown, no ```json fences
- Be concise and prioritise plant health
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_sensors",
            "description": "Get current greenhouse sensor readings",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_plant",
            "description": "Run vision model on current camera frame",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_crop_needs",
            "description": "Get optimal growing conditions for a crop",
            "parameters": {
                "type": "object",
                "properties": {
                    "crop": {"type": "string", "description": "crop name e.g. tomato"}
                },
                "required": ["crop"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_action",
            "description": "Log an actuator command — MUST be called every cycle",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {"type": "string", "enum": ["irrigation", "lights", "fan", "alert"]},
                    "value":       {"type": "string"},
                    "detail":      {"type": "string"},
                },
                "required": ["action_type", "value"],
            },
        },
    },
]

TOOL_FNS = {
    "read_sensors":      lambda args: read_sensors(),
    "analyze_plant":     lambda args: analyze_plant(),
    "lookup_crop_needs": lambda args: lookup_crop_needs(args.get("crop", "tomato")),
    "trigger_action":    lambda args: trigger_action(
                             args.get("action_type") or args.get("type") or "alert",
                             args.get("value") or args.get("action") or "triggered",
                             args.get("detail") or args.get("message") or ""),
}

STEP_LABELS = {
    "read_sensors":      "Reading sensors...",
    "analyze_plant":     "Analyzing plant image...",
    "lookup_crop_needs": "Looking up crop requirements...",
    "trigger_action":    "Triggering action...",
}

FALLBACK_ACTIONS = {
    "drought":    ("irrigation", "on",      "auto-triggered: moisture below threshold"),
    "disease":    ("alert",      "disease", "visual symptoms detected — apply fungicide, improve airflow"),
    "overwater":  ("fan",        "on",      "auto-triggered: moisture above threshold, improve drainage"),
    "heatstress": ("lights",     "reduced", "auto-triggered: temperature above threshold"),
    "normal":     ("alert",      "ok",      "all conditions within optimal range"),
}


def _parse_json(text: str) -> dict:
    """Robustly parse JSON from LLM output, stripping markdown fences if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]
    result = json.loads(text)
    # if assessment itself is a JSON string, double-parse it
    if isinstance(result.get("assessment"), str) and result["assessment"].strip().startswith("{"):
        try:
            result = json.loads(result["assessment"])
        except Exception:
            pass
    return result


def run_cycle(status_callback=None) -> dict:
    """Run one full assessment cycle."""

    # capture scenario at START of cycle before LLM takes time to reason
    initial_sensor_data = read_sensors()
    initial_scenario = initial_sensor_data.get("scenario", "normal")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "Run a greenhouse assessment cycle now."},
    ]

    action_was_triggered = False

    while True:
        response = nim_client.chat.completions.create(
            model="llama3.2",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1000,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            try:
                result = _parse_json(msg.content)
            except Exception:
                result = {
                    "assessment": msg.content.strip(),
                    "reasoning": "",
                    "actions_taken": [],
                }

            # fallback — if LLM never called trigger_action, force it using
            # the scenario captured at the START of the cycle
            if not action_was_triggered:
                if initial_scenario in FALLBACK_ACTIONS:
                    a_type, a_val, a_detail = FALLBACK_ACTIONS[initial_scenario]
                    trigger_action(a_type, a_val, a_detail)
                    existing = result.get("actions_taken") or []
                    result["actions_taken"] = existing + [f"{a_type}: {a_detail}"]

            return result

        # execute tool calls and feed results back into the loop
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            if name == "trigger_action":
                action_was_triggered = True
            if status_callback:
                status_callback(STEP_LABELS.get(name, f"Running {name}..."))
            args = json.loads(tc.function.arguments)
            result = TOOL_FNS[name](args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })


def run_loop(interval_seconds=30):
    print("SAGE agent loop running")
    while True:
        result = run_cycle()
        print(json.dumps(result, indent=2))
        time.sleep(interval_seconds)