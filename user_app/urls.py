from user_app import views
from django.urls import path 

urlpatterns = [
   path("",views.home,name="home"),
   path('video/', views.video, name='video'),
   path("viewrecords/",views.viewrecords,name="viewrecords"),
   path("register/", views.register_request, name="register"),
   path("login/", views.login_request, name="login"),
   path("traffics/",views.traffics,name='traffics'),
   
   
]
