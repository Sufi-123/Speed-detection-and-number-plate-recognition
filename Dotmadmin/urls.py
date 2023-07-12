from django import views
from django.contrib import admin
from django.urls import path , include
from Dotmadmin import views
urlpatterns = [
    path('', views.admin_login, name='dotm_login'),
    path("traffics/",views.traffics,name='traffics'),
    path ("welcome/",views.welcome_view,name='welcome_Dashboard'),
    path("chart/",views.chart_display,name='chart'),
    path ("notice/",views.notice,name='notice'),
    path("home/",views.dotm_home,name='home'),
]