from speed_estimation.vehicle_speed_count import process_video
from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse
from .models import Person , viewrecord,vehicle

# Create your views here.
def home (request):
    return render (request,'base.html')

# def viewrecords (request):
#     return render (request,'viewrecords.html')


from django.shortcuts import render
from django.db import connection


def video(request):
    return StreamingHttpResponse(process_video(), content_type='multipart/x-mixed-replace; boundary=frame')

def viewrecords(request):
    viewrecord_list= viewrecord.objects.all()
    return render ( request ,'viewrecords.html',
                   {'viewrecord_list': viewrecord_list})



def vehicle(request):
    vehicle_list= vehicle.objects.all()
    return render ( request ,'viewrecords.html',
                   {'vehicle_list': vehicle_list})




from django.shortcuts import  render, redirect
from .forms import NewUserForm
from django.contrib.auth import login
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, authenticate

def register_request(request):
	if request.method == "POST":
		form = NewUserForm(request.POST)
		if form.is_valid():
			user = form.save()
			login(request, user)
			messages.success(request, "Registration successful." )
			return redirect("/")
		messages.error(request, "Unsuccessful registration. Invalid information.")
	form = NewUserForm()
	return render (request=request, template_name="register.html", context={"register_form":form})


def login_request(request):
	if request.method == "POST":
		form = AuthenticationForm(request, data=request.POST)
		if form.is_valid():
			username = form.cleaned_data.get('username')
			password = form.cleaned_data.get('password')
			user = authenticate(username=username, password=password)
			if user is not None:
				login(request, user)
				messages.info(request, f"You are now logged in as {username}.")
				return redirect("/")
			else:
				messages.error(request,"Invalid username or password.")
		else:
			messages.error(request,"Invalid username or password.")
	form = AuthenticationForm()
	return render(request=request, template_name="login.html", context={"login_form":form})
# def viewrecords(request):

#     # with connection.cursor() as cursor:
#     #     cursor.execute("SELECT * FROM viewrecordss")
#     #     rows = cursor.fetchall()
#     # return render(request, 'viewrecords.html', {'rows': rows})

