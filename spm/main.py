"""
agent_api.py

Flask HTTP API wrapper around ProductivityAgent.

Exposes:
- /health                          (GET)
- /agent/json                      (POST)   --> generic JSON contract entry
- /agent/message                   (POST)   --> free-form text message
- /goals                           (POST)   --> create goal
- /goals/<goal_id>/progress        (PATCH)  --> update goal progress
- /goals                           (GET)    --> list goals for a user
- /reflections                     (POST)   --> add reflection
- /reminders                       (GET)    --> generate reminders
- /report                          (GET)    --> text productivity report
- /analysis                        (GET)    --> reflection analysis
- /accountability                  (GET)    --> accountability payload
- /insights                        (GET)    --> personalized insights

All responses are JSON.
"""
import os

from datetime import datetime
import logging

from flask import Flask, request, jsonify

from dotenv import load_dotenv


from productivity_agent import ProductivityAgent  # <-- make sure filename matches


# ------------------------------------------------------------------------------
# Basic app & logging setup
# ------------------------------------------------------------------------------

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False  # keep JSON keys in insertion order

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Simple Registry & Supervisor (for assignment structure)
# ------------------------------------------------------------------------------

class AgentRegistry:
    """
    Very simple registry that manages one named worker agent.
    You can extend this later to support multiple agents.
    """

    def __init__(self):
        self._agents = {}

    def get_agent(self, name: str = "productivity") -> ProductivityAgent:
        if name not in self._agents:
            self._agents[name] = ProductivityAgent()
        return self._agents[name]


class Supervisor:
    """
    Supervisor that receives high-level tasks and delegates to the agent.
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def health(self) -> dict:
        # For now just simple info; you can add DB checks, etc.
        now = datetime.utcnow().isoformat() + "Z"
        return {
            "status": "ok",
            "service": "ProductivityAgentAPI",
            "time_utc": now,
        }

    def handle_json_task(self, user_id: str, task: str, params: dict) -> dict:
        """
        Generic JSON contract handler. Delegates to the worker agent's
        handle_json_request(), which also uses LTM.
        """
        agent = self.registry.get_agent()
        request_payload = {
            "user_id": user_id,
            "task": task,
            "params": params or {},
        }
        return agent.handle_json_request(request_payload)


load_dotenv()  # Load .env file

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "productivity_agent")


registry = AgentRegistry()
supervisor = Supervisor(registry)


def get_agent() -> ProductivityAgent:
    """Convenience shortcut."""
    return registry.get_agent()


# ------------------------------------------------------------------------------
# Health check endpoint
# ------------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """
    Simple health check for logging & uptime monitoring.
    """
    logger.info("GET /health")
    data = supervisor.health()
    return jsonify(data), 200


# ------------------------------------------------------------------------------
# Generic JSON contract endpoint (good for Supervisor/Registry)
# ------------------------------------------------------------------------------

@app.route("/agent/json", methods=["POST"])
def agent_json():
    """
    Generic JSON-based entry point.

    Expects JSON:
    {
      "user_id": "affan",
      "task": "accountability" | "freeform_message" | "analyze_reflections" | ...,
      "params": { ... }
    }
    """
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400

    user_id = payload.get("user_id")
    task = payload.get("task")
    params = payload.get("params", {}) or {}

    if not user_id or not task:
        return jsonify({
            "status": "error",
            "message": "Missing 'user_id' or 'task' in request body."
        }), 400

    logger.info("POST /agent/json user=%s task=%s", user_id, task)

    try:
        result = supervisor.handle_json_task(user_id, task, params)
        return jsonify(result), 200
    except Exception as e:
        logger.exception("Error in /agent/json")
        return jsonify({"status": "error", "message": str(e)}), 500


# ------------------------------------------------------------------------------
# Free-form message endpoint (natural language input)
# ------------------------------------------------------------------------------

@app.route("/agent/message", methods=["POST"])
def agent_message():
    """
    User sends a single free-form message.
    Expects JSON:
    {
      "user_id": "affan",
      "message": "I want to finish my OS assignment by 2025-12-01, ..."
    }

    Uses LTM-aware handle_freeform_message() inside the agent.
    """
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400

    user_id = payload.get("user_id")
    message = payload.get("message", "")

    if not user_id or not message:
        return jsonify({
            "status": "error",
            "message": "Both 'user_id' and 'message' are required."
        }), 400

    logger.info("POST /agent/message user=%s", user_id)

    agent = get_agent()
    try:
        result = agent.handle_freeform_message(user_id, message)
        return jsonify(result), 200
    except Exception as e:
        logger.exception("Error in /agent/message")
        return jsonify({"status": "error", "message": str(e)}), 500


# ------------------------------------------------------------------------------
# Goal endpoints
# ------------------------------------------------------------------------------

@app.route("/goals", methods=["POST"])
def create_goal():
    """
    Create a goal (structured input).

    Expects JSON:
    {
      "user_id": "affan",
      "title": "Finish OS assignment",
      "description": "Complete all questions...",
      "category": "academic",
      "goal_type": "weekly",
      "deadline": "2025-12-01",
      "priority": 5
    }
    """
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400

    required = ["user_id", "title", "description", "category", "goal_type", "deadline", "priority"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({
            "status": "error",
            "message": f"Missing fields: {', '.join(missing)}"
        }), 400

    agent = get_agent()
    try:
        goal_id = agent.create_goal(
            user_id=data["user_id"],
            title=data["title"],
            description=data["description"],
            category=data["category"],
            goal_type=data["goal_type"],
            deadline=data["deadline"],
            priority=int(data["priority"]),
        )
        return jsonify({"status": "created", "goal_id": goal_id}), 201
    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        logger.exception("Error in /goals (POST)")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/goals", methods=["GET"])
def list_goals():
    """
    List all goals for a user.
    Query param: ?user_id=affan
    """
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"status": "error", "message": "Missing 'user_id' query parameter."}), 400

    agent = get_agent()
    try:
        goals = agent._get_goals_for_user(user_id)
        return jsonify({"status": "ok", "goals": goals}), 200
    except Exception as e:
        logger.exception("Error in /goals (GET)")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/goals/<goal_id>/progress", methods=["PATCH"])
def update_goal_progress(goal_id):
    """
    Update progress for a goal.

    Expects JSON:
    {
      "progress": 60
    }
    """
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400

    if "progress" not in data:
        return jsonify({"status": "error", "message": "Missing 'progress' in body."}), 400

    progress = data["progress"]
    agent = get_agent()
    try:
        msg = agent.update_goal_progress(goal_id, float(progress))
        return jsonify({"status": "ok", "message": msg}), 200
    except Exception as e:
        logger.exception("Error in /goals/<goal_id>/progress")
        return jsonify({"status": "error", "message": str(e)}), 500


# ------------------------------------------------------------------------------
# Reflection endpoints
# ------------------------------------------------------------------------------

@app.route("/reflections", methods=["POST"])
def add_reflection():
    """
    Add a reflection.

    Expects JSON:
    {
      "user_id": "affan",
      "date": "2025-12-01",          # optional, defaults to today if missing
      "text": "Today I felt stressed...",
      "achievements": "...",        # optional
      "obstacles": "..."            # optional
    }
    """
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400

    user_id = data.get("user_id")
    text = data.get("text")
    if not user_id or not text:
        return jsonify({
            "status": "error",
            "message": "Fields 'user_id' and 'text' are required."
        }), 400

    date = data.get("date") or datetime.now().strftime("%Y-%m-%d")
    achievements = data.get("achievements", "")
    obstacles = data.get("obstacles", "")

    agent = get_agent()
    try:
        rid = agent.add_reflection(
            user_id=user_id,
            date=date,
            text=text,
            achievements=achievements,
            obstacles=obstacles,
        )
        return jsonify({"status": "saved", "reflection_id": rid}), 201
    except Exception as e:
        logger.exception("Error in /reflections (POST)")
        return jsonify({"status": "error", "message": str(e)}), 500


# ------------------------------------------------------------------------------
# Analytics / Reports / Insights
# ------------------------------------------------------------------------------

@app.route("/reminders", methods=["GET"])
def get_reminders():
    """
    Get reminders for a user.
    Query param: ?user_id=affan
    """
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"status": "error", "message": "Missing 'user_id' query parameter."}), 400

    agent = get_agent()
    try:
        reminders = agent.generate_reminders(user_id)
        return jsonify({"status": "ok", "reminders": reminders}), 200
    except Exception as e:
        logger.exception("Error in /reminders (GET)")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/report", methods=["GET"])
def get_report():
    """
    Get text productivity report for a user.
    Query param: ?user_id=affan
    """
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"status": "error", "message": "Missing 'user_id' query parameter."}), 400

    agent = get_agent()
    try:
        report_text = agent.generate_text_report(user_id)
        return jsonify({"status": "ok", "report": report_text}), 200
    except Exception as e:
        logger.exception("Error in /report (GET)")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/analysis", methods=["GET"])
def get_analysis():
    """
    Reflection analysis (patterns, time-series, recommendations, anomalies).
    Query param: ?user_id=affan
    """
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"status": "error", "message": "Missing 'user_id' query parameter."}), 400

    agent = get_agent()
    try:
        analysis = agent.analyze_reflections(user_id)
        return jsonify({"status": "ok", "analysis": analysis}), 200
    except Exception as e:
        logger.exception("Error in /analysis (GET)")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/accountability", methods=["GET"])
def get_accountability():
    """
    Accountability payload for Supervisor Agent.
    Query param: ?user_id=affan
    """
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"status": "error", "message": "Missing 'user_id' query parameter."}), 400

    agent = get_agent()
    try:
        payload = agent.get_accountability_payload(user_id)
        return jsonify({"status": "ok", "payload": payload}), 200
    except Exception as e:
        logger.exception("Error in /accountability (GET)")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/insights", methods=["GET"])
def get_insights():
    """
    Personalized insights (best weekday, average completion by category, etc.).
    Query param: ?user_id=affan
    """
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"status": "error", "message": "Missing 'user_id' query parameter."}), 400

    agent = get_agent()
    try:
        insights = agent.get_personalized_insights(user_id)
        return jsonify({"status": "ok", "insights": insights}), 200
    except Exception as e:
        logger.exception("Error in /insights (GET)")
        return jsonify({"status": "error", "message": str(e)}), 500


# ------------------------------------------------------------------------------
# Run server
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # You can change host/port as you like.
    # debug=True is handy for development; turn it off in production.
    app.run(host="0.0.0.0", port=8000, debug=True)
