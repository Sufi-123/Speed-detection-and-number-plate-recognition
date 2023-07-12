from django import views
from django.contrib import admin
from django.urls import path , include
from .views import *
urlpatterns = [
    path('', admin_login, name='dotm_login'),
    path("register/", register_station, name="registration"),
    path("traffics/",traffics,name='traffics'),
    path ("welcome/",welcome_view,name='welcome_Dashboard'),
    path("chart/",chart_display,name='chart'),
    path ("notice/",notice,name='notice'),
    path("home/",dotm_home,name='home'),
]