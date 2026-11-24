from django.urls import path
from . import views

urlpatterns = [
    path("", views.user_login, name="login"),
    path("register/", views.register, name="register"),
    path("logout/", views.user_logout, name="logout"),
    path("home/", views.home, name="home"),
    path("chatbot/", views.chatbot, name="chatbot"),
    path("chatbot/clear/", views.clear_chat, name="clear_chat"),
]
