from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
import joblib
import numpy as np

# Load trained model
model = joblib.load("heart_model.pkl")


# ---------------- LOGIN ----------------
def user_login(request):
    if request.method == "POST":
        user = authenticate(
            request,
            username=request.POST.get("username"),
            password=request.POST.get("password")
        )
        if user:
            login(request, user)
            return redirect("/home/")
    return render(request, "login.html")


# ---------------- REGISTER ----------------
def register(request):
    if request.method == "POST":
        User.objects.create_user(
            username=request.POST.get("username"),
            password=request.POST.get("password")
        )
        return redirect("/")
    return render(request, "register.html")


# ---------------- LOGOUT ----------------
def user_logout(request):
    logout(request)
    return redirect("/")


# ---------------- HOME / PREDICTION ----------------
@login_required
def home(request):
    result = None
    risk = None
    risk_level = "success"
    doctor = None
    hospital = None

    if request.method == "POST":
        try:
            data = [
                int(request.POST.get("age")),
                int(request.POST.get("sex")),
                int(request.POST.get("cp")),
                int(request.POST.get("trestbps")),
                int(request.POST.get("chol")),
                int(request.POST.get("fbs")),
                int(request.POST.get("restecg")),
                int(request.POST.get("thalach")),
                int(request.POST.get("exang")),
                float(request.POST.get("oldpeak")),
                int(request.POST.get("slope")),
                int(request.POST.get("ca")),
                int(request.POST.get("thal")),
            ]

            prediction = model.predict([data])[0]
            probability = model.predict_proba([data])[0][1]

            risk = round(probability * 100, 2)

            if risk > 60:
                risk_level = "danger"
                doctor = "Dr. Rajesh Kumar (Cardiologist)"
                hospital = "Apollo Heart Institute"
            elif risk > 30:
                risk_level = "warning"
                doctor = "Dr. Ananya Rao (Cardiac Specialist)"
                hospital = "Fortis Cardiac Care"
            else:
                risk_level = "success"
                doctor = "Dr. General Physician"
                hospital = "Nearby Primary Health Center"

            result = (
                "Heart Disease Detected"
                if prediction == 1
                else "No Heart Disease Detected"
            )

        except Exception as e:
            result = "Invalid Input"
            risk = "N/A"
            print("Prediction Error:", e)

    return render(request, "home.html", {
        "result": result,
        "risk": risk,
        "risk_level": risk_level,
        "doctor": doctor,
        "hospital": hospital,
    })


# ---------------- CHATBOT ----------------
@login_required
def chatbot(request):
    reply = ""

    if request.method == "POST":
        msg = request.POST.get("message", "").lower()

        if not msg:
            reply = "Please type a question."
        elif "bp" in msg or "blood pressure" in msg:
            reply = "Normal blood pressure is around 120/80 mm Hg."
        elif "cholesterol" in msg:
            reply = "Cholesterol should be below 200 mg/dL."
        elif "exercise" in msg:
            reply = "30 minutes of daily exercise improves heart health."
        elif "heart" in msg:
            reply = "Heart disease risk depends on age, lifestyle, BP and cholesterol."
        else:
            reply = "I can help with BP, cholesterol, exercise and heart health."

    return render(request, "chatbot.html", {"reply": reply})
