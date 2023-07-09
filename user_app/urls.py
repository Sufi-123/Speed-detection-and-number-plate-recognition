from user_app import views
from django.urls import path 
from .views import download_csv
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
   path('download-csv/', download_csv, name='download_csv'),
   path("",views.home,name="home"),
   path('video/', views.video, name='video'),
   path("viewrecords/",views.viewrecords,name="viewrecords"),
   path("register/", views.register_request, name="register"),
   path("login/", views.login_request, name="login"),
   path("traffics/",views.traffics,name='traffics'),
   path ("welcome/",views.welcome_view,name='welcome_Dashboard'),
   path('chart/', views.view_records, name='chart')
   
   
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
