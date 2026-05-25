from django.urls import path
from .views import *

urlpatterns = [
    path("", base, name="base"),
    path("about/",about ,name="about"),
    path("fees/",fees ,name="fees"),
    path("assignment/",assignment ,name="assignment"),
    path("attendance/",attendance ,name="attendance"),
    path("login/",login ,name="login"),
    path("notice/",notice ,name="notice"),
    path("profile/",profile ,name="profile"),
    path("register/",register ,name="register"),
    path("result/",result ,name="result"),
    path("student/",student ,name="student"),
    path("timetable/",timetable ,name="timetable")
]