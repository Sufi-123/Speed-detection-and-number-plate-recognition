from user_app import views
from django.urls import path 

urlpatterns = [
    path("",views.welcome_page, name="welcome"),
    path('download-csv/', views.download_csv, name='download_csv'),
   path("Home/",views.home,name="home"),
   path('video/', views.video, name='video'),
   path("Records/",views.Records,name="Records"),
]
