from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse


class AuthFlowTests(TestCase):
    def test_protected_routes_redirect_to_login_root(self):
        for route_name in ("home", "chatbot", "clear_chat"):
            response = self.client.get(reverse(route_name))
            self.assertRedirects(response, f"/?next={reverse(route_name)}")

    def test_register_validates_and_rejects_duplicate(self):
        short_pw = self.client.post(
            reverse("register"),
            {"username": "alice", "password": "123"},
        )
        self.assertEqual(short_pw.status_code, 200)
        self.assertContains(short_pw, "at least 6 characters")

        response = self.client.post(
            reverse("register"),
            {"username": "alice", "password": "strongpassword123"},
        )
        self.assertRedirects(response, reverse("login"))
        self.assertTrue(User.objects.filter(username="alice").exists())

        duplicate = self.client.post(
            reverse("register"),
            {"username": "alice", "password": "strongpassword123"},
        )
        self.assertEqual(duplicate.status_code, 200)
        self.assertContains(duplicate, "Username already exists.")

    def test_login_invalid_shows_error_and_valid_redirects_home(self):
        User.objects.create_user(username="bob", password="secretpassword")

        bad = self.client.post(
            reverse("login"),
            {"username": "bob", "password": "wrong"},
        )
        self.assertEqual(bad.status_code, 200)
        self.assertContains(bad, "Invalid username or password.")

        good = self.client.post(
            reverse("login"),
            {"username": "bob", "password": "secretpassword"},
        )
        self.assertRedirects(good, reverse("home"))

        out = self.client.get(reverse("logout"))
        self.assertRedirects(out, reverse("login"))


class PredictionAndChatbotTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="tester", password="password123")
        self.client.login(username="tester", password="password123")

    def test_full_clinical_prediction(self):
        payload = {
            "mode": "clinical",
            "age": "63",
            "sex": "1",
            "cp": "0",
            "trestbps": "145",
            "chol": "233",
            "fbs": "1",
            "restecg": "2",
            "thalch": "150",
            "exang": "0",
            "oldpeak": "2.3",
            "slope": "2",
            "ca": "0",
            "thal": "2",
        }
        response = self.client.post(reverse("home"), payload)
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.context["risk"], float)
        self.assertEqual(response.context["provided_count"], 13)
        self.assertEqual(response.context["imputed_labels"], [])
        self.assertContains(response, "Medical Screening Disclaimer")

    def test_guided_mode_imputes_unknown_clinical_fields(self):
        payload = {
            "mode": "guided",
            "age": "58",
            "sex": "1",
            "cp": "3",
            "exang": "1",
            "trestbps": "",
            "chol": "",
            "fbs": "unknown",
            "restecg": "unknown",
            "thalch": "",
            "oldpeak": "",
            "slope": "unknown",
            "ca": "unknown",
            "thal": "unknown",
        }
        response = self.client.post(reverse("home"), payload)
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.context["risk"], float)
        self.assertEqual(response.context["provided_count"], 4)
        self.assertEqual(len(response.context["imputed_labels"]), 9)
        self.assertContains(response, "Imputed from clinical dataset baseline")

    def test_out_of_range_input_returns_validation_error(self):
        payload = {
            "age": "150",
            "sex": "1",
            "cp": "0",
            "exang": "0",
            "trestbps": "10",
        }
        response = self.client.post(reverse("home"), payload)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.context["risk"])
        self.assertTrue(len(response.context["validation_errors"]) >= 2)
        self.assertContains(response, "Age must be between 1 and 120.")
        self.assertContains(response, "Resting Blood Pressure must be between 50 and 260.")

    def test_chatbot_persists_history_and_clears(self):
        resp1 = self.client.post(reverse("chatbot"), {"message": "What is normal blood pressure?"})
        self.assertEqual(resp1.status_code, 200)
        self.assertContains(resp1, "120/80 mm Hg")
        self.assertIsNotNone(resp1.context["timestamp"])
        self.assertIsNotNone(resp1.context["bot_time"])

        resp2 = self.client.post(reverse("chatbot"), {"message": "Explain oldpeak and thal"})
        self.assertEqual(resp2.status_code, 200)
        self.assertContains(resp2, "120/80 mm Hg")
        self.assertContains(resp2, "Oldpeak")
        self.assertEqual(len(resp2.context["chat_history"]), 4)

        resp3 = self.client.post(reverse("chatbot"), {"message": "Tips for diet and sleep"})
        self.assertEqual(resp3.status_code, 200)
        self.assertContains(resp3, "7–8 hours")
        self.assertEqual(len(resp3.context["chat_history"]), 6)

        cleared = self.client.get(reverse("clear_chat"), follow=True)
        self.assertRedirects(cleared, reverse("chatbot"))
        self.assertEqual(cleared.context["chat_history"], [])
