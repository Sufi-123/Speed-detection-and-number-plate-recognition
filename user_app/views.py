# from speed_estimation.vehicle_speed_count import process_video
from speed_estimation.combined import process_video
from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse
from .models import  viewrecord,traffic
import csv

# Create your views here.
def home (request):
    viewrecord_list= viewrecord.objects.all()
    return render (request,'base.html',{'viewrecord_list':viewrecord_list})

# def viewrecords (request):
#     return render (request,'viewrecords.html')


from django.shortcuts import render
from django.db import connection


def video(request):
    return StreamingHttpResponse(process_video(), content_type='multipart/x-mixed-replace; boundary=frame')

def viewrecords(request):
    viewrecord_list= viewrecord.objects.all()
    context = {
        'viewrecord_list': viewrecord_list
    }
    return render(request, 'viewrecords.html', context)

def download_csv(request):
    # Retrieve data from the database or any other source
    records = viewrecord.objects.all()  # Fetch records from the ViewRecord model

    # Create a response object with CSV content
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="view_records.csv"'

    # Create a CSV writer and write the header row
    writer = csv.writer(response)
    writer.writerow(['SN', 'License Plate No', 'Speed', 'Date', 'ID', 'Count'])

    # Write the data rows
    for record in records:
        writer.writerow([
            record.pk,
            record.liscenceplate_no,
            record.speed,
            record.date,
            record.IDs,
            record.count
        ])

    return response
    



def welcome_view(request):
	return render(request,'welcome_dashboard.html')





def traffics(request):
    traffic_list= traffic.objects.all()
    if request.method == 'POST':
        TrafficBooth = request.POST['trafficBooth']
        Areacode = request.POST['sn']
        location = request.POST['areaCode']
		

        traffic_booth = traffic(TrafficBooth=TrafficBooth, Areacode=Areacode, location=location)
        traffic_booth.save()
	

        return redirect('traffics')  # Replace 'traffic_list' with the URL name of your traffic list view

    return render(request, 'TrafficList.html',
		  {'traffic_list':traffic_list})  # Replace 'your_template.html' with the actual template file name


# def trafficlist(request):
# 	data = traffic(TrafficBooth=traffic,)
#     data.save()


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

