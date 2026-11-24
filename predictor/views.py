from datetime import datetime
import os

from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import redirect, render
import joblib
import numpy as np

MODEL_PATH = settings.BASE_DIR / "heart_model.pkl"
_model = None


def get_model():
    """Load and cache the trained heart disease screening pipeline."""
    global _model
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    return _model


HEART_FIELD_SPECS = [
    {"name": "age", "aliases": (), "label": "Age", "caster": int, "required": True, "min": 1, "max": 120},
    {"name": "sex", "aliases": (), "label": "Biological Sex", "caster": int, "required": True, "min": 0, "max": 1},
    {"name": "cp", "aliases": (), "label": "Chest Pain Type", "caster": int, "required": True, "min": 0, "max": 3},
    {"name": "trestbps", "aliases": (), "label": "Resting Blood Pressure", "caster": float, "required": False, "min": 50, "max": 260},
    {"name": "chol", "aliases": (), "label": "Serum Cholesterol", "caster": float, "required": False, "min": 50, "max": 700},
    {"name": "fbs", "aliases": (), "label": "Fasting Blood Sugar", "caster": float, "required": False, "min": 0, "max": 1},
    {"name": "restecg", "aliases": (), "label": "Resting ECG", "caster": float, "required": False, "min": 0, "max": 2},
    {"name": "thalch", "aliases": ("thalach",), "label": "Maximum Heart Rate (thalch)", "caster": float, "required": False, "min": 40, "max": 250},
    {"name": "exang", "aliases": (), "label": "Exercise-Induced Chest Pain", "caster": int, "required": True, "min": 0, "max": 1},
    {"name": "oldpeak", "aliases": (), "label": "ST Depression (Oldpeak)", "caster": float, "required": False, "min": -3.0, "max": 10.0},
    {"name": "slope", "aliases": (), "label": "ST Segment Slope", "caster": float, "required": False, "min": 0, "max": 2},
    {"name": "ca", "aliases": (), "label": "Major Vessels (CA)", "caster": float, "required": False, "min": 0, "max": 3},
    {"name": "thal", "aliases": (), "label": "Thalassemia Test (Thal)", "caster": float, "required": False, "min": 1, "max": 3},
]

HEALTH_DISCLAIMER = (
    "Note: This information is a general health-screening aid, not a medical diagnosis."
)

CHATBOT_RESPONSES = [
    (
        ("bp", "blood pressure", "hypertension", "systolic", "diastolic"),
        "Normal resting blood pressure is around 120/80 mm Hg. Readings consistently at or above 140/90 mm Hg should be reviewed by a doctor. Reducing excess salt, staying active, and managing stress help support healthy pressure.",
    ),
    (
        ("cholesterol", "chol", "lipid", "ldl", "hdl", "triglyceride"),
        "Total cholesterol is generally healthiest below 200 mg/dL. High LDL can narrow heart arteries over time. Eating oats, beans, vegetables, and healthy fats while limiting fried or processed foods helps lower cholesterol.",
    ),
    (
        ("exercise", "walk", "workout", "activity", "fitness"),
        "Aim for about 30 minutes of moderate physical activity (such as brisk walking) on most days, unless a doctor advises rest. Stop and seek medical advice if activity triggers chest tightness, dizziness, or unusual shortness of breath.",
    ),
    (
        ("chest", "pain", "angina", "tightness", "emergency", "breath", "sweat"),
        "Typical angina feels like chest pressure or tightness during exertion and eases with rest, whereas non-anginal pain is often sharper or unrelated to effort. Seek emergency medical care immediately for sudden chest pressure, pain spreading to the arm/jaw/back, cold sweats, or shortness of breath.",
    ),
    (
        ("ecg", "restecg", "ekg", "oldpeak", "st depression", "slope", "thal", "thalassemia", "ca", "vessel", "fluoroscopy", "thalch", "thalach", "heart rate", "fbs", "blood sugar", "glucose"),
        "In plain terms: Resting ECG checks your heart's electrical rhythm at rest; Oldpeak and Slope measure stress-ECG changes during exercise; Max Heart Rate (thalch) is peak pulse on a treadmill test; CA counts major blood vessels seen on a scan; Thal checks blood flow on a heart scan; and Fasting Blood Sugar checks morning glucose. If you lack any of these reports, pick 'Unknown / No report'.",
    ),
    (
        ("diet", "food", "salt", "sodium", "nutrition", "sleep", "stress", "smoke", "smoking"),
        "Core heart-protective habits include eating vegetables, fruits, whole grains, and lean proteins; limiting salt and sugary drinks; avoiding tobacco; getting 7–8 hours of regular sleep; and practicing daily stress-relief.",
    ),
    (
        ("unknown", "report", "missing", "test", "screen", "risk", "guided"),
        "If you do not have ECG, stress test, or lab reports, choose 'Unknown / No report' (or leave optional numbers blank). The screening model fills missing clinical fields from trained baselines and displays your data completeness score.",
    ),
    (
        ("heart", "disease"),
        "Heart disease risk reflects a combination of age, blood pressure, cholesterol, blood sugar, chest symptoms, and daily habits. Use the Screening Dashboard for an early risk estimate and follow up with a clinician.",
    ),
]


def _get_rule_based_reply(message: str) -> str:
    msg = message.strip().lower()
    if not msg:
        return "Please type a health question."
    for keywords, response in CHATBOT_RESPONSES:
        if any(keyword in msg for keyword in keywords):
            return f"{response} {HEALTH_DISCLAIMER}"
    return (
        "I can explain blood pressure, cholesterol, chest symptoms, exercise, diet/sleep/stress basics, "
        "and screening terms like ECG, oldpeak, thal, or ca in plain language. "
        f"{HEALTH_DISCLAIMER}"
    )


def _get_llm_reply(message: str, history: list) -> str | None:
    """Optionally generate a dynamic answer via Gemini if GEMINI_API_KEY is configured."""
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None

    try:
        from google import genai
    except ImportError:
        return None

    recent_turns = "\n".join(
        f"{item.get('sender', 'user').capitalize()}: {item.get('text', '')}"
        for item in history[-6:]
    )
    prompt = (
        f"Recent conversation:\n{recent_turns}\n\nUser: {message}"
        if recent_turns
        else message
    )
    system_instruction = (
        "You are a concise, empathetic heart-health screening assistant. "
        "Explain medical terms (BP, cholesterol, chest pain, ECG, oldpeak, thal, ca) in plain language. "
        "Never diagnose or prescribe. Remind users this is general screening guidance and to seek "
        "emergency care for severe chest pain or shortness of breath."
    )

    try:
        client = genai.Client(api_key=api_key)
        interaction = client.interactions.create(
            model="gemini-4-hydrogen",
            input=prompt,
            system_instruction=system_instruction,
        )
        text = (interaction.output_text or "").strip()
        if text:
            return text
    except Exception:
        return None
    return None


def _get_chatbot_reply(message: str, history: list | None = None) -> str:
    llm_reply = _get_llm_reply(message, history or [])
    if llm_reply:
        return llm_reply
    return _get_rule_based_reply(message)


def _read_field_raw(post_data, spec) -> str:
    raw = post_data.get(spec["name"])
    if raw is None:
        for alias in spec["aliases"]:
            raw = post_data.get(alias)
            if raw is not None:
                break
    return (raw or "").strip()


def _parse_heart_inputs(post_data):
    """Validate and parse 13 clinical/guided fields, returning NaN for unknown optional fields."""
    values = []
    imputed_labels = []
    errors = []
    form_values = {}

    for spec in HEART_FIELD_SPECS:
        raw = _read_field_raw(post_data, spec)
        form_values[spec["name"]] = raw

        is_unknown = raw == "" or raw.lower() == "unknown"
        if is_unknown:
            if spec["required"]:
                errors.append(f"{spec['label']} is required.")
                values.append(np.nan)
            else:
                values.append(np.nan)
                imputed_labels.append(spec["label"])
            continue

        try:
            parsed = spec["caster"](raw)
        except (TypeError, ValueError):
            errors.append(f"{spec['label']} must be a valid number.")
            values.append(np.nan)
            continue

        if parsed < spec["min"] or parsed > spec["max"]:
            errors.append(
                f"{spec['label']} must be between {spec['min']} and {spec['max']}."
            )
            values.append(np.nan)
            continue

        values.append(float(parsed))

    return np.array([values], dtype=float), imputed_labels, errors, form_values


# ---------------- LOGIN ----------------
def user_login(request):
    if request.user.is_authenticated:
        return redirect("home")

    error = None
    if request.method == "POST":
        username = (request.POST.get("username") or "").strip()
        password = request.POST.get("password") or ""
        if not username or not password:
            error = "Username and password are required."
        else:
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect("home")
            error = "Invalid username or password."

    return render(request, "login.html", {"error": error})


# ---------------- REGISTER ----------------
def register(request):
    if request.user.is_authenticated:
        return redirect("home")

    error = None
    if request.method == "POST":
        username = (request.POST.get("username") or "").strip()
        password = request.POST.get("password") or ""

        if not username or not password:
            error = "Username and password are required."
        elif len(password) < 6:
            error = "Password must be at least 6 characters long."
        elif User.objects.filter(username=username).exists():
            error = "Username already exists."
        else:
            User.objects.create_user(username=username, password=password)
            return redirect("login")

    return render(request, "register.html", {"error": error})


# ---------------- LOGOUT ----------------
def user_logout(request):
    logout(request)
    return redirect("login")


# ---------------- HOME / SCREENING DASHBOARD ----------------
@login_required
def home(request):
    mode = request.POST.get("mode", "guided") if request.method == "POST" else "guided"
    if mode not in ("guided", "clinical"):
        mode = "guided"

    context = {
        "mode": mode,
        "result": None,
        "risk": None,
        "risk_level": "success",
        "risk_label": None,
        "doctor": None,
        "hospital": None,
        "guidance": None,
        "imputed_labels": [],
        "provided_count": 0,
        "total_fields": len(HEART_FIELD_SPECS),
        "completeness_pct": 0,
        "validation_errors": [],
        "form_values": {},
    }

    if request.method == "POST":
        X, imputed_labels, errors, form_values = _parse_heart_inputs(request.POST)
        context["form_values"] = form_values

        if errors:
            context["validation_errors"] = errors
            return render(request, "home.html", context)

        model = get_model()
        if model is None:
            context["validation_errors"] = [
                f"Trained model file not found at {MODEL_PATH}. Run train_heart.py first."
            ]
            return render(request, "home.html", context)

        probability = float(model.predict_proba(X)[0][1])
        risk = round(probability * 100, 2)
        prediction = 1 if probability >= 0.5 else 0

        if risk > 60:
            risk_level = "danger"
            risk_label = "High Screening Risk"
            doctor = "Cardiologist Consultation Recommended"
            hospital = "Specialty Cardiac Care Center"
            guidance = "Please arrange a prompt medical evaluation with ECG and blood work."
        elif risk > 30:
            risk_level = "warning"
            risk_label = "Moderate Screening Risk"
            doctor = "Primary Care / Cardiac Specialist"
            hospital = "Community Hospital or Clinic"
            guidance = "Schedule a checkup to review blood pressure, cholesterol, and heart tests."
        else:
            risk_level = "success"
            risk_label = "Low Screening Risk"
            doctor = "General Physician (Routine Checkup)"
            hospital = "Primary Health Center"
            guidance = "Maintain heart-healthy habits and routine preventive checkups."

        provided_count = len(HEART_FIELD_SPECS) - len(imputed_labels)
        completeness_pct = round((provided_count / len(HEART_FIELD_SPECS)) * 100)

        context.update({
            "result": (
                "Elevated Heart Disease Risk Indicators"
                if prediction == 1
                else "Lower Heart Disease Risk Indicators"
            ),
            "risk": risk,
            "risk_level": risk_level,
            "risk_label": risk_label,
            "doctor": doctor,
            "hospital": hospital,
            "guidance": guidance,
            "imputed_labels": imputed_labels,
            "provided_count": provided_count,
            "completeness_pct": completeness_pct,
        })

    return render(request, "home.html", context)


# ---------------- CHATBOT ----------------
@login_required
def chatbot(request):
    history = list(request.session.get("chat_history", []))
    timestamp = None
    bot_time = None

    if request.method == "POST":
        user_message = (request.POST.get("message") or "").strip()
        if user_message:
            timestamp = datetime.now().strftime("%H:%M")
            reply = _get_chatbot_reply(user_message, history)
            bot_time = datetime.now().strftime("%H:%M")
            history.append({"sender": "user", "text": user_message, "time": timestamp})
            history.append({"sender": "bot", "text": reply, "time": bot_time})
            request.session["chat_history"] = history
            request.session.modified = True
    elif history:
        for item in reversed(history):
            if item.get("sender") == "user" and timestamp is None:
                timestamp = item.get("time")
            elif item.get("sender") == "bot" and bot_time is None:
                bot_time = item.get("time")
            if timestamp and bot_time:
                break

    return render(
        request,
        "chatbot.html",
        {
            "chat_history": history,
            "timestamp": timestamp,
            "bot_time": bot_time,
        },
    )


@login_required
def clear_chat(request):
    request.session["chat_history"] = []
    request.session.modified = True
    return redirect("chatbot")
