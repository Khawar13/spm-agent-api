# Upgraded ProductivityAgent (drop-in replacement for your current class)

import os

from bson import ObjectId
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import re
from collections import defaultdict, Counter
from dateutil import parser as dateparser  # more flexible date parsing if available
import json
from pymongo import MongoClient


DATE_FORMAT = "%Y-%m-%d"


class ProductivityAgent:
    """
    Upgraded ProductivityAgent with:
    - better NLP (lemmatization if available)
    - sentiment (VADER if available, else small fallback)
    - weighted scoring & intensifiers
    - trend detection (time series)
    - goal risk prediction (ETA vs deadline)
    - personalized insights (weekday productivity, category speed)
    - goal-aware reflection mapping
    - anomaly detection (missing streaks, spikes)
    """


    def __init__(self, connection_string: Optional[str] = None, db_name: str = "productivity_agent"):
        # Use provided connection_string, else read from env
        if connection_string is None:
            connection_string = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        db_name = os.getenv("MONGO_DB_NAME", db_name)
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]

        self.goals_collection = self.db.goals
        self.reflections_collection = self.db.reflections
        self.ltm_collection = self.db.ltm_entries   # 👈 NEW
        self._init_db()

        # NLP resources (attempt to import optional libs)
        self._init_text_processing()

        # base keyword dictionary (you can expand more)
        self.base_keywords = {
            "stressed": ["stress", "stressed", "anxious", "anxiety", "overwhelmed", "pressure", "panic", "worried",
                         "tension", "frustrated", "burnout", "burnt out", "tense", "nervous", "emotional"],
            "tired": ["tired", "exhausted", "fatigued", "sleepy", "drained", "sleep deprived", "lethargic"],
            "focused": ["focused", "productive", "in the zone", "concentrated", "on track", "engaged", "motivated"],
            "procrastination": ["procrastinate", "procrastination", "delayed", "postponed", "avoid", "distracted",
                                "social media", "binge watched", "time wasting"],
            "happy": ["happy", "joy", "joyful", "excited", "satisfied", "content", "grateful", "proud"],
            "sad": ["sad", "upset", "depressed", "lonely", "down", "unhappy"],
            "health": ["headache", "sick", "ill", "fever", "cold", "cough", "pain", "weakness"]
        }

        # intensifiers and downtoners (weights)
        # presence of an intensifier near a keyword multiplies the base weight
        self.intensifiers = {
            "very": 1.8, "extremely": 2.0, "super": 1.6, "really": 1.4, "so": 1.3, "terribly": 1.9
        }
        self.downtoners = {"slightly": 0.6, "a little": 0.7, "somewhat": 0.8, "bit": 0.8}

        # default category weights (you can tune)
        self.category_weight = {
            "procrastination": 1.3,
            "stressed": 1.2,
            "tired": 1.1,
            "focused": 0.9,
            "happy": 0.7,
            "sad": 1.0,
            "health": 1.2
        }

    # ---------- DB helpers ----------
    def _init_db(self):
        self.goals_collection.create_index("user_id")
        self.goals_collection.create_index([("user_id", 1), ("deadline", 1)])
        self.reflections_collection.create_index("user_id")
        self.reflections_collection.create_index([("user_id", 1), ("date", 1)])
        # for LTM
        self.ltm_collection.create_index("user_id")
        self.ltm_collection.create_index([("user_id", 1), ("input_type", 1), ("input_key", 1)], unique=True)

    # ---------- NLP / Text processor ----------
    def _init_text_processing(self):
        # Try to set up lemmatizer / wordnet
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            from nltk.corpus import wordnet
            nltk_available = True
            try:
                # ensure wordnet is available (user should have downloaded)
                wordnet.ensure_loaded()
            except Exception:
                pass
            self.lemmatizer = WordNetLemmatizer()
            self._use_lemmatizer = True
        except Exception:
            # fallback to trivial stemmer (strip common suffixes)
            self._use_lemmatizer = False
            self.lemmatizer = None

        # attempt to import WordNet synonyms expansion
        try:
            from nltk.corpus import wordnet as wn
            self._wn = wn
            self._use_wordnet = True
        except Exception:
            self._wn = None
            self._use_wordnet = False

        # Attempt VADER sentiment
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.sent_analyzer = SentimentIntensityAnalyzer()
            self._use_vader = True
        except Exception:
            self.sent_analyzer = None
            self._use_vader = False
            # fallback sentiment lexicon
            self._pos_words = set(["good", "great", "happy", "productive", "energized", "motivated", "focused"])
            self._neg_words = set(["bad", "sad", "stressed", "tired", "exhausted", "anxious", "depressed", "procrastinate"])

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        # replace fancy apostrophes
        text = text.replace("’", "'")
        # remove extra punctuation except spaces (keep words)
        text = re.sub(r"[^a-z0-9'\s]", " ", text)
        # collapse spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokens(self, text: str) -> List[str]:
        t = self._normalize_text(text)
        if self._use_lemmatizer:
            # basic tokenization then lemmatize
            tokens = re.findall(r"[a-z']+", t)
            return [self.lemmatizer.lemmatize(tok) for tok in tokens]
        else:
            # simple tokens
            return re.findall(r"[a-z']+", t)
        
    def handle_json_request(self, payload: dict) -> dict:
        """
        Handles generic JSON requests from /agent/json.
        """
        user_id = payload.get("user_id")
        task = payload.get("task")
        params = payload.get("params", {})

        if task == "freeform_message":
            message = params.get("message", "")
            if not message:
                return {"status": "error", "message": "Missing 'message' in params."}
            return self.handle_freeform_message(user_id, message)

        elif task == "accountability":
            return {"status": "ok", "payload": self.get_accountability_payload(user_id)}

        elif task == "analyze_reflections":
            return {"status": "ok", "analysis": self.analyze_reflections(user_id)}

        elif task == "generate_report":
            return {"status": "ok", "report": self.generate_text_report(user_id)}

        elif task == "get_insights":
            return {"status": "ok", "insights": self.get_personalized_insights(user_id)}
        
        elif task == "goal":
            # Extract goal fields from params
            title = params.get("title")
            description = params.get("description")
            category = params.get("category")
            goal_type = params.get("goal_type")
            deadline = params.get("deadline")
            priority = params.get("priority")

            # Validate required fields
            if not all([title, description, category, goal_type, deadline, priority]):
                return {"status": "error", "message": "Missing required goal fields in params."}

            # Create goal via agent method
            goal_id = self.create_goal(
                user_id=user_id,
                title=title,
                description=description,
                category=category,
                goal_type=goal_type,
                deadline=deadline,
                priority=priority
            )

            return {
                "status": "created",
                "goal_id": goal_id,
                "type": "goal",
                "used_data": params
            }


        else:
            return {"status": "error", "message": f"Unknown task '{task}'."}
        

    def _expand_with_wordnet(self, word: str) -> List[str]:
        # return synonyms if wordnet available, else empty
        if not self._use_wordnet:
            return []
        syns = set()
        for syn in self._wn.synsets(word):
            for lem in syn.lemmas():
                syns.add(lem.name().replace("_", " ").lower())
        return list(syns)
    
    def interpret_message_type(self, text: str) -> str:
        """
        Very simple classifier:
        - 'goal'        → for messages that look like goal creation
        - 'progress'    → for progress updates
        - 'reflection'  → default
        """
        t = text.lower()

        # goal-ish words
        if any(w in t for w in ["goal:", "create goal", "set a goal", "my goal is", "i want to", "i need to"]):
            return "goal"

        # mentions deadline / priority / type words = also likely a goal
        if any(w in t for w in ["deadline", "by ", "before ", "due ", "priority", "daily", "weekly", "long-term"]):
            return "goal"

        # progress-style messages
        if any(w in t for w in ["update progress", "% done", "percent done", "completed", "finished", "i am at", "progress to"]):
            return "progress"

        # fallback
        return "reflection"

    def _semantic_keyword_set(self, keywords: Dict[str, List[str]]) -> Dict[str, set]:
        """
        Expand keyword lists by synonyms where possible.
        """
        expanded = {}
        for cat, words in keywords.items():
            s = set()
            for w in words:
                s.add(w)
                # add lemma form
                for tok in self._tokens(w):
                    s.add(tok)
                # expand synonyms
                if self._use_wordnet:
                    for syn in self._expand_with_wordnet(w):
                        s.add(syn)
                        for tok in self._tokens(syn):
                            s.add(tok)
            expanded[cat] = s
        return expanded

    def _sentiment_score(self, text: str) -> Dict[str, float]:
        """
        Return sentiment dict: {'neg':..., 'neu':..., 'pos':..., 'compound':...}
        Uses VADER if available, else a simple lexicon-based fallback.
        """
        if self._use_vader:
            return self.sent_analyzer.polarity_scores(text)
        else:
            # naive fallback: count positives/negatives
            tokens = set(self._tokens(text))
            pos = len(tokens & self._pos_words)
            neg = len(tokens & self._neg_words)
            total = max(1, pos + neg)
            compound = (pos - neg) / total
            return {"neg": float(neg / total), "neu": 1.0 - abs(compound), "pos": float(pos / total), "compound": compound}

    # ---------- Old helpers kept ----------
    def _parse_date(self, s: str) -> datetime:
        # more flexible parsing using dateutil if present
        try:
            return datetime.strptime(s, DATE_FORMAT)
        except Exception:
            try:
                return dateparser.parse(s)
            except Exception:
                raise
    def parse_goal_from_message(self, message: str) -> Dict[str, Any]:
        """
        Try to extract a goal from a free-form message.
        Returns dict:
        {
          "ok": bool,
          "missing_fields": [...],
          "data": {title, description, category, goal_type, deadline, priority}
        }
        """
        text = message.strip()
        low = text.lower()
        missing = []
        data: Dict[str, Any] = {}

        # 1) deadline: look for YYYY-MM-DD
        m_date = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if m_date:
            data["deadline"] = m_date.group(1)
        else:
            missing.append("deadline (format YYYY-MM-DD)")

        # 2) category
        category = None
        for cat in ["academic", "personal", "professional"]:
            if cat in low:
                category = cat
                break
        if category:
            data["category"] = category
        else:
            missing.append("category (academic / personal / professional)")

        # 3) goal_type
        goal_type = None
        if "daily" in low:
            goal_type = "daily"
        elif "weekly" in low:
            goal_type = "weekly"
        elif "long-term" in low or "long term" in low:
            goal_type = "long-term"
        if goal_type:
            data["goal_type"] = goal_type
        else:
            missing.append("goal_type (daily / weekly / long-term)")

        # 4) priority (look for 'priority X')
        m_prio = re.search(r"priority\s*([1-5])", low)
        if m_prio:
            data["priority"] = int(m_prio.group(1))
        else:
            missing.append("priority (1–5, e.g. 'priority 4')")

        # 5) title & description (very simple heuristic)
        # If user wrote "title:" use that. Else take first sentence as title, rest as description
        m_title = re.search(r"title\s*:\s*(.+)", text, flags=re.IGNORECASE)
        if m_title:
            title = m_title.group(1).strip()
            data["title"] = title
            # description = everything minus that line
            data["description"] = text
        else:
            # split by '.' and take first sentence as title
            parts = [p.strip() for p in re.split(r"[.!?]", text) if p.strip()]
            if parts:
                data["title"] = parts[0][:80]  # cap length
                data["description"] = text
            else:
                missing.append("title/description (what the goal actually is)")

        ok = len(missing) == 0
        return {"ok": ok, "missing_fields": missing, "data": data}
    
    def parse_reflection_from_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """
        Interpret a free-form text as a reflection.
        Checks if user talked about:
          - feelings/mood
          - achievements
          - obstacles
        If something is missing, we mark it.

        Returns:
        {
          "ok": bool,
          "missing_parts": [...],
          "reflection_doc": {...}  # ready for add_reflection if ok
        }
        """
        text = message.strip()
        low = text.lower()
        missing_parts = []

        # 1) simple mood detection using some keywords
        mood_words = ["happy", "sad", "stressed", "tired", "anxious", "overwhelmed",
                      "excited", "relaxed", "angry", "frustrated"]
        has_mood = any(w in low for w in mood_words)
        if not has_mood:
            missing_parts.append("how you felt (e.g. stressed, tired, happy, etc.)")

        # 2) achievements detection
        ach_patterns = ["finished", "completed", "managed to", "i did", "i got done",
                        "i achieved", "i solved", "i worked on", "i studied"]
        has_achievement = any(p in low for p in ach_patterns)
        if not has_achievement:
            missing_parts.append("at least one achievement (something you actually did)")

        # 3) obstacles detection
        obst_patterns = ["couldn't", "could not", "stuck", "blocked", "distracted",
                         "procrastinated", "didn't have time", "no time", "too tired",
                         "hard to focus", "interrupted"]
        has_obstacle = any(p in low for p in obst_patterns)
        if not has_obstacle:
            missing_parts.append("obstacles or what got in your way")

        # date = today by default
        today = datetime.now().strftime(DATE_FORMAT)

        reflection_doc = {
            "user_id": user_id,
            "date": today,
            "text": message,
            "achievements": "",  # you can optionally auto-extract later
            "obstacles": ""
        }

        ok = len(missing_parts) == 0
        return {
            "ok": ok,
            "missing_parts": missing_parts,
            "reflection_doc": reflection_doc
        }

    def _get_goals_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        goals_cursor = self.goals_collection.find({"user_id": user_id})
        goals = []
        for goal in goals_cursor:
            goals.append({
                "id": str(goal["_id"]),
                "title": goal.get("title", ""),
                "category": goal.get("category", ""),
                "goal_type": goal.get("goal_type", ""),
                "deadline": goal.get("deadline", ""),
                "priority": goal.get("priority", 0),
                "progress": goal.get("progress", 0),
                "status": goal.get("status", "pending"),
                "created_at": goal.get("created_at", ""),
                "completed_at": goal.get("completed_at"),
            })
        return goals

    # ---------- New: goal risk prediction ----------
    def compute_goal_risk(self, user_id: str, goal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict risk of missing a goal by comparing expected completion date from historical speed.
        Returns: {'risk': 'low'/'medium'/'high', 'eta': 'YYYY-MM-DD' or None, 'days_to_deadline': int}
        """
        # compute average completion speed for this user (days per percent)
        metrics = self.compute_performance_metrics(user_id, days_window=365)
        # fallback if no completion_times data: use 1% per day default (i.e., 100 days to finish)
        avg_days = metrics.get("average_completion_days")
        if avg_days is None:
            avg_days_per_goal = 7.0  # assume one week average
        else:
            avg_days_per_goal = max(1.0, avg_days)

        # compute remaining percentage and estimate days
        current_progress = goal.get("progress", 0)
        remaining_pct = max(0.0, 100.0 - current_progress)
        # crude speed: percent/day = 100 / avg_days_per_goal
        pct_per_day = 100.0 / avg_days_per_goal
        if pct_per_day <= 0:
            est_days = None
        else:
            est_days = remaining_pct / pct_per_day

        created_at = None
        try:
            created_at = self._parse_date(goal.get("created_at"))
        except Exception:
            created_at = None

        now = datetime.now()
        deadline_dt = None
        try:
            deadline_dt = self._parse_date(goal.get("deadline"))
        except Exception:
            # if invalid deadline, return medium risk
            return {"risk": "medium", "eta": None, "days_to_deadline": None}

        eta_dt = (now + timedelta(days=est_days)) if est_days is not None else None
        days_to_deadline = (deadline_dt - now).days
        days_until_eta = (eta_dt - now).days if eta_dt else None

        # risk logic (tunable)
        risk = "low"
        if eta_dt is None:
            risk = "medium"
        else:
            if eta_dt > deadline_dt:
                risk = "high"
            elif (deadline_dt - eta_dt).days < 2:
                risk = "medium"
            else:
                risk = "low"

        return {"risk": risk, "eta": eta_dt.strftime(DATE_FORMAT) if eta_dt else None, "days_to_deadline": days_to_deadline}

    # ---------- New: map reflections to goals ----------
    def map_reflections_to_goals(self, user_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Attempt to link reflections to goals based on keyword overlap between
        goal title/description and reflection text (plus synonyms).
        Returns mapping {goal_id: [reflections...]}
        """
        goals = self._get_goals_for_user(user_id)
        reflections = self._get_reflections_for_user(user_id)
        # prepare semantic tokens for reflections
        ref_texts = [(r["date"], r["text"], r) for r in reflections]
        mapping = defaultdict(list)
        # expand goal tokens
        for g in goals:
            goal_tokens = set(self._tokens(g.get("title", "") + " " + g.get("category", "") + " " + g.get("description", "")))
            # add wordnet synonyms if available for each token
            if self._use_wordnet:
                extra = set()
                for tok in list(goal_tokens):
                    for syn in self._expand_with_wordnet(tok):
                        extra.update(self._tokens(syn))
                goal_tokens |= extra

            for date, text, r in ref_texts:
                tokens = set(self._tokens(text))
                overlap = len(goal_tokens & tokens)
                # threshold: at least one overlapping token OR category token in text
                if overlap >= 1 or g.get("category", "") in text.lower():
                    mapping[g["id"]].append(r)
        return mapping

    # ---------- Enhanced analyze_reflections ----------
    def analyze_reflections(self, user_id: str) -> Dict[str, Any]:
        """
        Enhanced reflection analysis:
        - weighted pattern counts
        - time-series per-day per-category
        - sentiment aggregation
        - anomalies (missing streaks, spikes)
        - mapping to goals
        """
        reflections = self._get_reflections_for_user(user_id)
        if not reflections:
            return {"message": "No reflections found."}

        # prepare semantic keyword sets
        expanded = self._semantic_keyword_set(self.base_keywords)

        # results accumulators
        weighted_counts = {k: 0.0 for k in expanded}
        time_series = defaultdict(lambda: {k: 0.0 for k in expanded})
        sentiment_series = defaultdict(list)

        # helper window for intensifier detection: check word +/-3 tokens for intensifier or downtoner
        for r in reflections:
            date = r.get("date", "")
            text = r.get("text", "")
            normalized = self._normalize_text(text)
            tokens = self._tokens(text)
            token_str = " ".join(tokens)

            # per-reflection sentiment
            sent = self._sentiment_score(text)
            sentiment_series[date].append(sent)

            # for each category, compute weighted occurrences
            for cat, kws in expanded.items():
                cat_score = 0.0
                for kw in kws:
                    # exact phrase match more valuable
                    if kw in token_str:
                        # base weight
                        w = 1.0
                        # look for intensifiers/downtoners close to the keyword
                        # simple proximity search: check a small window around the word in normalized text
                        pattern = r"(?:\b(?:{})\b)".format(re.escape(kw))
                        for m in re.finditer(pattern, token_str):
                            start, end = m.start(), m.end()
                            # window tokens
                            # find words before the match (up to 3)
                            before = token_str[max(0, start - 40):start].split()
                            after = token_str[end:end + 40].split()
                            context = before[-3:] + after[:3]
                            # check intensifiers
                            inten_mult = 1.0
                            for ctxt in context:
                                if ctxt in self.intensifiers:
                                    inten_mult *= self.intensifiers[ctxt]
                                if ctxt in self.downtoners:
                                    inten_mult *= self.downtoners[ctxt]
                                # simple negation handling
                                if ctxt in ("not", "n't", "never"):
                                    inten_mult *= -1.0
                            cat_score += w * inten_mult
                    # also check token-wise equivalence (lemmatized tokens)
                    elif kw in tokens:
                        w = 0.8
                        cat_score += w
                # apply category-level weight
                cat_score *= self.category_weight.get(cat, 1.0)
                if cat_score != 0.0:
                    weighted_counts[cat] += cat_score
                    time_series[date][cat] += cat_score

        # aggregate sentiment per day into a simple average compound
        aggregated_sentiment = {}
        for date, arr in sentiment_series.items():
            compounds = [a.get("compound", 0.0) for a in arr]
            aggregated_sentiment[date] = sum(compounds) / max(1, len(compounds))

        # build trend detection (compare last 7 days vs previous 7 days)
        dates_sorted = sorted(time_series.keys())
        trend_notes = []
        if dates_sorted:
            # flatten last 7 and previous 7
            def sum_range(days_list):
                agg = defaultdict(float)
                for d in days_list:
                    for k, v in time_series[d].items():
                        agg[k] += v
                return agg

            last7 = dates_sorted[-7:]
            prev7 = dates_sorted[-14:-7] if len(dates_sorted) >= 14 else dates_sorted[:-7]
            last7_agg = sum_range(last7) if last7 else {}
            prev7_agg = sum_range(prev7) if prev7 else {}

            for cat in expanded:
                l = last7_agg.get(cat, 0.0)
                p = prev7_agg.get(cat, 0.0)
                if p == 0 and l > 0:
                    trend_notes.append(f"{cat} appeared recently.")
                elif p > 0:
                    change = (l - p) / p
                    if change > 0.3:
                        trend_notes.append(f"{cat} is increasing ({change:.0%} increase).")
                    elif change < -0.3:
                        trend_notes.append(f"{cat} is decreasing ({abs(change):.0%} drop).")

        # anomaly detection: missing reflections streaks and stress spikes
        anomalies = []
        # missing streaks: check last N days presence
        try:
            all_dates = sorted({r["date"] for r in reflections})
            if all_dates:
                last_date = datetime.strptime(all_dates[-1], DATE_FORMAT)
                streak_missing = 0
                # look back 7 days to see if multiple days missing
                for i in range(1, 8):
                    d = (last_date - timedelta(days=i)).strftime(DATE_FORMAT)
                    if d not in all_dates:
                        streak_missing += 1
                if streak_missing >= 3:
                    anomalies.append(f"Missing reflections for {streak_missing} of the last 7 days.")
        except Exception:
            pass

        # stress spikes: compare last day stress score vs median
        stress_values = [time_series[d].get("stressed", 0.0) for d in time_series]
        if stress_values:
            median = sorted(stress_values)[len(stress_values) // 2]
            last_day = dates_sorted[-1] if dates_sorted else None
            if last_day and time_series[last_day].get("stressed", 0.0) > 2 * (median + 1e-6):
                anomalies.append("Recent stress spike detected on last reflection day.")

        # mapping reflections to goals
        goal_map = self.map_reflections_to_goals(user_id)

        # recommendations based on weighted_counts
        recommendations = []
        if weighted_counts.get("stressed", 0) > 0:
            recommendations.append(
                "You often mention feeling stressed. Consider scheduling smaller, realistic goals and short breaks."
            )
        if weighted_counts.get("tired", 0) > 0:
            recommendations.append(
                "Tiredness shows up — improving sleep schedule or short naps could help productivity."
            )
        if weighted_counts.get("procrastination", 0) > 0:
            recommendations.append(
                "Procrastination is present. Use time-boxing (Pomodoro) or start with a 5-minute warm-up task."
            )
        if weighted_counts.get("focused", 0) > 0:
            recommendations.append(
                "You have focused days. Try to replicate the environment or routine from those days."
            )
        if weighted_counts.get("happy", 0) > 0:
            recommendations.append("Good mood detected; keep tracking what works.")
        if weighted_counts.get("sad", 0) > 0:
            recommendations.append("You mention sadness; consider social support or rest if it persists.")
        if weighted_counts.get("health", 0) > 0:
            recommendations.append("Health issues mentioned — prioritize rest and consult care if needed.")

        if not recommendations:
            recommendations.append("No strong patterns detected. Keep reflecting regularly.")

        return {
            "pattern_counts": weighted_counts,
            "time_series": {d: time_series[d] for d in sorted(time_series)},
            "sentiment_by_date": aggregated_sentiment,
            "goal_reflection_map": goal_map,
            "recommendations": recommendations,
            "trend_notes": trend_notes,
            "anomalies": anomalies,
        }

    # ---------- Performance analytics (unchanged but used by compute_goal_risk) ----------
    def compute_performance_metrics(self, user_id: str, days_window: int = 30) -> Dict[str, Any]:
        goals = self._get_goals_for_user(user_id)
        if not goals:
            return {"message": "No goals found for this user."}

        now = datetime.now()
        window_start = now - timedelta(days=days_window)

        total = 0
        completed = 0
        missed = 0
        in_progress = 0

        completion_times = []  # in days

        recent_completions = 0
        earlier_completions = 0

        for g in goals:
            try:
                created_at = self._parse_date(g["created_at"])
            except Exception:
                created_at = now
            status = g["status"]
            total += 1

            if status == "completed":
                completed += 1
                if g["completed_at"]:
                    try:
                        completed_at = self._parse_date(g["completed_at"])
                        delta = completed_at - created_at
                        completion_times.append(delta.days if delta.days >= 0 else 0)
                        if completed_at >= window_start:
                            recent_completions += 1
                        else:
                            earlier_completions += 1
                    except Exception:
                        pass
            elif status == "missed":
                missed += 1
            elif status == "in-progress":
                in_progress += 1

        completion_rate = (completed / total) * 100 if total > 0 else 0
        avg_completion_days = (
            sum(completion_times) / len(completion_times) if completion_times else None
        )

        trend = "stable"
        if recent_completions > earlier_completions:
            trend = "improving"
        elif recent_completions < earlier_completions:
            trend = "declining"

        efficiency_scores = []
        for g in goals:
            if g["status"] == "completed" and g["completed_at"]:
                try:
                    deadline = self._parse_date(g["deadline"])
                    completed_at = self._parse_date(g["completed_at"])
                    days_before_deadline = (deadline - completed_at).days
                    efficiency_scores.append(days_before_deadline)
                except Exception:
                    pass
        avg_efficiency = (
            sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else None
        )

        return {
            "total_goals": total,
            "completed_goals": completed,
            "missed_goals": missed,
            "in_progress_goals": in_progress,
            "completion_rate": round(completion_rate, 2),
            "average_completion_days": avg_completion_days,
            "productivity_trend": trend,
            "average_days_before_deadline": avg_efficiency,
            "recent_completions_last_30_days": recent_completions,
        }

    # ---------- Other existing utilities ----------
    def create_goal(
        self,
        user_id: str,
        title: str,
        description: str,
        category: str,
        goal_type: str,
        deadline: str,
        priority: int,
    ) -> str:
        if category not in ['academic', 'personal', 'professional']:
            raise ValueError(f"Invalid category: {category}")
        if goal_type not in ['daily', 'weekly', 'long-term']:
            raise ValueError(f"Invalid goal_type: {goal_type}")
        if not (1 <= priority <= 5):
            raise ValueError(f"Priority must be between 1 and 5, got: {priority}")

        created_at = datetime.now().strftime(DATE_FORMAT)
        goal_doc = {
            "user_id": user_id,
            "title": title,
            "description": description,
            "category": category,
            "goal_type": goal_type,
            "deadline": deadline,
            "priority": priority,
            "progress": 0.0,
            "status": "pending",
            "created_at": created_at,
            "completed_at": None
        }

        result = self.goals_collection.insert_one(goal_doc)
        return str(result.inserted_id)

    def update_goal_progress(self, goal_id: str, progress: float) -> str:
        if progress < 0:
            progress = 0
        if progress > 100:
            progress = 100

        goal = self.goals_collection.find_one({"_id": ObjectId(goal_id)})
        if not goal:
            return f"Goal {goal_id} not found."

        old_progress = goal.get("progress", 0)
        status = goal.get("status", "pending")

        new_status = status
        completed_at = None
        message = ""

        if progress == 100:
            new_status = "completed"
            completed_at = datetime.now().strftime(DATE_FORMAT)
            message = (
                "🎉 Goal completed! Great job! "
                "Take a moment to reflect on what worked well and log a reflection entry."
            )
        elif progress > 0:
            if status in ("pending", "missed"):
                new_status = "in-progress"

        update_data = {"progress": progress, "status": new_status}
        if completed_at:
            update_data["completed_at"] = completed_at

        self.goals_collection.update_one({"_id": ObjectId(goal_id)}, {"$set": update_data})

        if not message:
            if progress > old_progress:
                message = "✅ Progress updated. Keep going!"
            elif progress < old_progress:
                message = "Progress reduced. Make sure this reflects reality; adjust your plan if needed."
            else:
                message = "No change in progress."

        return message

    def add_reflection(
        self,
        user_id: str,
        date: str,
        text: str,
        achievements: str = "",
        obstacles: str = "",
    ) -> str:
        reflection_doc = {
            "user_id": user_id,
            "date": date,
            "text": text,
            "achievements": achievements,
            "obstacles": obstacles
        }
        result = self.reflections_collection.insert_one(reflection_doc)
        return str(result.inserted_id)

    def _get_reflections_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        reflections_cursor = self.reflections_collection.find({"user_id": user_id}).sort("date", 1)
        reflections = []
        for reflection in reflections_cursor:
            reflections.append({
                "date": reflection.get("date", ""),
                "text": reflection.get("text", ""),
                "achievements": reflection.get("achievements", ""),
                "obstacles": reflection.get("obstacles", ""),
            })
        return reflections

    def generate_reminders(self, user_id: str, now: Optional[datetime] = None) -> List[str]:
        now = now or datetime.now()
        goals = self._get_goals_for_user(user_id)
        reminders = []

        for g in goals:
            try:
                deadline = self._parse_date(g["deadline"])
            except Exception:
                continue
            status = g["status"]
            progress = g["progress"]
            days_diff = (deadline - now).days

            if status != "completed" and deadline < now:
                reminders.append(
                    f"⚠️ You missed the deadline for '{g['title']}'. Review what blocked you and reschedule or adjust the goal."
                )
                self.goals_collection.update_one(
                    {"_id": ObjectId(g["id"]), "status": {"$ne": "completed"}},
                    {"$set": {"status": "missed"}}
                )
            elif status in ("pending", "in-progress") and 0 <= days_diff <= 1:
                urgency = "today" if days_diff == 0 else "tomorrow"
                reminders.append(
                    f"⏰ Reminder: '{g['title']}' is due {urgency}. Current progress: {progress:.0f}%. Try to make a bit of progress now."
                )

        if not reminders:
            reminders.append("🎯 No urgent or missed goals right now. Nice work staying on top of things!")

        return reminders

    def generate_text_report(self, user_id: str) -> str:
        metrics = self.compute_performance_metrics(user_id)
        if "message" in metrics:
            return metrics["message"]

        lines = []
        lines.append("=== Productivity Report ===")
        lines.append(f"Total goals: {metrics['total_goals']}")
        lines.append(f"Completed: {metrics['completed_goals']}")
        lines.append(f"In-progress: {metrics['in_progress_goals']}")
        lines.append(f"Missed: {metrics['missed_goals']}")
        lines.append(f"Completion rate: {metrics['completion_rate']}%")
        if metrics["average_completion_days"] is not None:
            lines.append(f"Average time to complete a goal: {metrics['average_completion_days']:.1f} days")
        if metrics["average_days_before_deadline"] is not None:
            lines.append(
                f"On average, you finish goals {metrics['average_days_before_deadline']:.1f} days "
                "before the deadline (positive is good, negative = after deadline)."
            )
        lines.append(f"Productivity trend (last 30 days): {metrics['productivity_trend']}")
        lines.append(f"Goals completed in the last 30 days: {metrics['recent_completions_last_30_days']}")
        return "\n".join(lines)

    def get_accountability_payload(self, user_id: str) -> Dict[str, Any]:
        metrics = self.compute_performance_metrics(user_id)
        reflections_analysis = self.analyze_reflections(user_id)
        # add goal risk for each active goal
        goals = self._get_goals_for_user(user_id)
        goal_risks = {}
        for g in goals:
            goal_risks[g["id"]] = self.compute_goal_risk(user_id, g)
        payload = {
            "user_id": user_id,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "performance_metrics": metrics,
            "reflection_summary": reflections_analysis,
            "goal_risks": goal_risks,
        }
        return payload

    # ---------- New: personalized insights ----------
    def get_personalized_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Compute insights like:
        - weekday productivity (which weekday has most completions)
        - average completion days by category
        """
        goals = self._get_goals_for_user(user_id)
        if not goals:
            return {"message": "No goals found."}

        by_weekday = defaultdict(int)
        completions_by_category = defaultdict(list)

        for g in goals:
            if g["status"] == "completed" and g.get("completed_at"):
                try:
                    created = self._parse_date(g["created_at"])
                    completed = self._parse_date(g["completed_at"])
                    weekday = completed.strftime("%A")
                    by_weekday[weekday] += 1
                    # completion days
                    days = (completed - created).days
                    completions_by_category[g.get("category", "unknown")].append(days)
                except Exception:
                    pass

        # find most productive weekday
        if by_weekday:
            best_day = max(by_weekday.items(), key=lambda x: x[1])[0]
        else:
            best_day = None

        avg_completion_by_cat = {cat: (sum(lst)/len(lst) if lst else None) for cat, lst in completions_by_category.items()}

        return {
            "best_weekday": best_day,
            "completions_by_weekday": dict(by_weekday),
            "avg_completion_days_by_category": avg_completion_by_cat
        }

    def handle_freeform_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """
        High-level entry:
        - Detect type (goal / reflection / progress)
        - Use LTM first
        - If not in LTM, run normal logic and store result
        """

        # 1) build an LTM key from the message
        #    you can also include interpreted message type later if you want
        ltm_payload = {"message": message}
        input_key = self._make_ltm_key(ltm_payload)

        # 2) check LTM
        cached = self._ltm_lookup(user_id, input_type="freeform_message", input_key=input_key)
        if cached is not None:
            return cached

        # 3) normal flow (your original logic below)

        msg_type = self.interpret_message_type(message)

        if msg_type == "goal":
            parsed = self.parse_goal_from_message(message)
            if parsed["ok"]:
                d = parsed["data"]
                goal_id = self.create_goal(
                    user_id=user_id,
                    title=d["title"],
                    description=d["description"],
                    category=d["category"],
                    goal_type=d["goal_type"],
                    deadline=d["deadline"],
                    priority=d["priority"],
                )
                response = {
                    "type": "goal",
                    "status": "created",
                    "goal_id": goal_id,
                    "used_data": d,
                }
            else:
                response = {
                    "type": "goal",
                    "status": "incomplete",
                    "missing_fields": parsed["missing_fields"],
                    "message": "Please include the missing pieces in your next message.",
                }

        elif msg_type == "reflection":
            parsed = self.parse_reflection_from_message(user_id, message)
            if parsed["ok"]:
                rid = self.add_reflection(
                    user_id=user_id,
                    date=parsed["reflection_doc"]["date"],
                    text=parsed["reflection_doc"]["text"],
                    achievements=parsed["reflection_doc"]["achievements"],
                    obstacles=parsed["reflection_doc"]["obstacles"],
                )
                response = {
                    "type": "reflection",
                    "status": "saved",
                    "reflection_id": rid,
                    "missing_parts": [],
                }
            else:
                response = {
                    "type": "reflection",
                    "status": "incomplete",
                    "missing_parts": parsed["missing_parts"],
                    "message": "Your reflection is missing some parts. Please also talk about: "
                               + "; ".join(parsed["missing_parts"]),
                }

        else:
            # TODO: you can implement progress parsing later
            response = {
                "type": msg_type,
                "status": "unsupported_yet",
                "message": "Detected as a progress/update message, but parser not implemented yet.",
            }

        # 4) store successful response in LTM
        self._ltm_store(user_id, input_type="freeform_message", input_key=input_key, response=response)

        # 5) return fresh response
        return {**response, "_source": "fresh"}

   # ---------- LTM (Long-Term Memory) Helpers ----------

    def _make_ltm_key(self, payload: dict) -> str:
        """
        Build a stable string key from a request payload.
        We sort keys so the same content always gives the same key.
        """
        try:
            return json.dumps(payload, sort_keys=True)
        except Exception:
            return str(payload)

    def _ltm_lookup(self, user_id: str, input_type: str, input_key: str) -> Optional[Dict[str, Any]]:
        """
        Check LTM: if we have a stored response for this (user, type, key).
        """
        doc = self.ltm_collection.find_one({
            "user_id": user_id,
            "input_type": input_type,
            "input_key": input_key,
        })
        if doc:
            # optional: increment usage counter
            self.ltm_collection.update_one(
                {"_id": doc["_id"]},
                {"$inc": {"usage_count": 1}, "$set": {"last_used_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}}
            )
            # stored response is under "response"
            resp = doc.get("response", {})
            # mark that this came from memory (not required, but nice for debugging)
            if isinstance(resp, dict):
                resp = {**resp, "_source": "ltm"}
            return resp
        return None

    def _ltm_store(self, user_id: str, input_type: str, input_key: str, response: Dict[str, Any]) -> None:
        """
        Store a successful response in LTM for future reuse.
        """
        doc = {
            "user_id": user_id,
            "input_type": input_type,
            "input_key": input_key,
            "response": response,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "usage_count": 0,
        }
        # upsert: in case same key already exists
        self.ltm_collection.update_one(
            {"user_id": user_id, "input_type": input_type, "input_key": input_key},
            {"$set": doc},
            upsert=True
        )

from pprint import pprint

def main():
    # 1. Init agent & demo user
    agent = ProductivityAgent()
    user = "affan_demo"

    print("=== ProductivityAgent Demo for user:", user, "===")

    # 2. Clear old data for a clean run
    agent.goals_collection.delete_many({"user_id": user})
    agent.reflections_collection.delete_many({"user_id": user})
    print("[*] Cleared old goals and reflections for this user.\n")

    # 3. Create some goals directly (using structured inputs)
    print(">>> Creating goals (direct API calls)...")
    g1 = agent.create_goal(
        user_id=user,
        title="Finish OS assignment",
        description="Complete all OS questions and submit on LMS.",
        category="academic",
        goal_type="weekly",
        deadline=(datetime.now() + timedelta(days=3)).strftime(DATE_FORMAT),
        priority=5,
    )

    g2 = agent.create_goal(
        user_id=user,
        title="Go to the gym",
        description="45 minutes of weight training.",
        category="personal",
        goal_type="daily",
        deadline=(datetime.now() + timedelta(days=1)).strftime(DATE_FORMAT),
        priority=3,
    )

    print(f"Created goal g1: {g1}")
    print(f"Created goal g2: {g2}\n")

    # 4. Create a goal via a single free-form message
    print(">>> Creating a goal via free-form message...")
    freeform_goal_msg = (
        "I want to update my CV and LinkedIn by "
        f"{(datetime.now() + timedelta(days=14)).strftime(DATE_FORMAT)}. "
        "It's a professional long-term goal, priority 4. "
        "I want to add my latest projects and courses."
    )

    res_goal_ff = agent.handle_freeform_message(user, freeform_goal_msg)
    pprint(res_goal_ff)
    print()

    # 5. Update goal progress
    print(">>> Updating goal progress for g1 and g2...")
    print("g1 -> 40%:", agent.update_goal_progress(g1, 40))
    print("g1 -> 80%:", agent.update_goal_progress(g1, 80))
    print("g1 -> 100%:", agent.update_goal_progress(g1, 100))
    print("g2 -> 100%:", agent.update_goal_progress(g2, 100))
    print()

    # 6. Add reflections directly
    print(">>> Adding reflections (direct)...")
    yesterday = (datetime.now() - timedelta(days=1)).strftime(DATE_FORMAT)
    today = datetime.now().strftime(DATE_FORMAT)

    r1 = agent.add_reflection(
        user_id=user,
        date=yesterday,
        text="Felt stressed and procrastinated on my assignment, but managed to outline the solution.",
        achievements="Outlined the solution.",
        obstacles="Got distracted by social media."
    )

    r2 = agent.add_reflection(
        user_id=user,
        date=today,
        text="Had a very focused study session and felt productive, but a bit tired at night.",
        achievements="Solved 4 difficult OS problems.",
        obstacles="Felt sleepy after dinner."
    )

    print(f"Added reflection r1: {r1}")
    print(f"Added reflection r2: {r2}\n")

    # 7. Add a reflection via free-form (single message)
    print(">>> Adding reflection via free-form message...")
    freeform_reflection_msg = (
        "Today I was very tired and stressed. "
        "I procrastinated a lot in the morning, but I managed to finish two problems in the evening."
    )
    res_ref_ff = agent.handle_freeform_message(user, freeform_reflection_msg)
    pprint(res_ref_ff)
    print()

    # 8. Generate reminders
    print("=== Reminders ===")
    reminders = agent.generate_reminders(user)
    for msg in reminders:
        print("-", msg)
    print()

    # 9. Text productivity report
    print("=== Productivity Report ===")
    print(agent.generate_text_report(user))
    print()

    # 10. Reflection analysis (advanced)
    print("=== Reflection Analysis ===")
    reflection_analysis = agent.analyze_reflections(user)
    pprint(reflection_analysis)
    print()

    # 11. Goal → risk prediction and mapping reflections to goals
    print("=== Goal Risks ===")
    goals = agent._get_goals_for_user(user)
    for g in goals:
        risk = agent.compute_goal_risk(user, g)
        print(f"- {g['title']} (status={g['status']}, progress={g['progress']}%) ->", risk)
    print()

    print("=== Reflection ↔ Goal Mapping ===")
    goal_map = agent.map_reflections_to_goals(user)
    pprint(goal_map)
    print()

    # 12. Accountability payload (for Supervisor Agent)
    print("=== Accountability Payload ===")
    payload = agent.get_accountability_payload(user)
    pprint(payload)
    print()

    # 13. Personalized insights (weekday/productivity, category speed)
    print("=== Personalized Insights ===")
    insights = agent.get_personalized_insights(user)
    pprint(insights)
    print()

    print("=== Demo complete. ===")

    # 14. LTM caching test (same message twice should hit memory)
    print("=== LTM test (freeform_message) ===")
    ltm_msg = (
        "I want to update my CV and LinkedIn by 2025-12-04. "
        "It's a professional long-term goal, priority 4. "
        "I want to add my latest projects and courses."
    )

    first = agent.handle_freeform_message(user, ltm_msg)
    second = agent.handle_freeform_message(user, ltm_msg)

    print("First call source:", first.get("_source"))
    print("Second call source:", second.get("_source"))



if __name__ == "__main__":
    main()
