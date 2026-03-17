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
4. Call trigger_action for each intervention needed
5. Return ONLY a valid JSON object with no markdown, no code fences, no extra text:
{"assessment": "one sentence plain-English plant status", "reasoning": "2-3 sentences explaining your decisions", "actions_taken": ["irrigation on 8min", "alert: heat stress detected"]}

Be concise. Prioritise plant health. Always explain your reasoning.
IMPORTANT: Your final response must be pure JSON only. No markdown. No ```json fences.
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
            "description": "Log an actuator command",
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

def _parse_json(text: str) -> dict:
    """Robustly parse JSON from LLM output, stripping markdown fences if present."""
    text = text.strip()
    # strip ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # find first { to last } in case there's surrounding text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]
    return json.loads(text)


def run_cycle(status_callback=None) -> dict:
    """Run one full assessment cycle. Optionally call status_callback(str) on each tool call."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "Run a greenhouse assessment cycle now."},
    ]

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
                return _parse_json(msg.content)
            except Exception:
                return {
                    "assessment": msg.content.strip(),
                    "reasoning": "",
                    "actions_taken": [],
                }

        # notify dashboard of current step
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
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