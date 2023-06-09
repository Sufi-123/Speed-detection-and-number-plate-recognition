from user_app import views
from django.urls import path 

urlpatterns = [
   path("",views.home,name="home"),
   path("viewrecords/",views.viewrecords,name="viewrecords")
   
]
