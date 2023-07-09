# from speed_estimation.vehicle_speed_count import process_video
from speed_estimation.combined import process_video
from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse
from .models import  viewrecord,traffic
import csv
from collections import defaultdict
from .visualize import generate_bar_graph, generate_line_graph,generate_permonth_graph,generate_perday_graph



def view_records(request):
    limit = 50  # Speed limit value
    records = viewrecord.objects.order_by('speed')[:20][::-1]  # Get the bottom 20 records by speed
    new_record= viewrecord.objects.order_by('speed')
    # Aggregate the number of vehicles per day
    day_count = defaultdict(int)
    for record in new_record:
        if record.speed > limit:
            day_count[record.date.strftime('%d/%m/%Y')] += 1

    # Extract the day-month-year labels and count values
    label1 = list(day_count.keys())
    count1 = list(day_count.values())


    # Aggregate the number of vehicles exceeding the speed limit per month
    month_count = defaultdict(int)
    for record in new_record:
        if record.speed > limit:
            month_count[record.date.strftime('%m/%Y')] += 1

    # Extract the month-year labels and count values
    labels = list(month_count.keys())
    counts = list(month_count.values())

    exceeded_limit = [record for record in records if record.speed > limit]
    within_limit = [record for record in records if record.speed <= limit]
    speeds = [record.speed for record in records]

    # Generate the line graph
    graph_path=generate_line_graph(speeds)
    chart_path = generate_bar_graph(exceeded_limit, within_limit)
    permonth_path= generate_permonth_graph(labels, counts)
    perday_path= generate_perday_graph(label1, count1)

    context={
        'chart_path': chart_path,
        'graph_path': graph_path,
        'permonth_path' : permonth_path,
        'perday_path' : perday_path
          

    }
    return render(request, 'chart.html',context)


# Create your views here.
def home (request):
    viewrecord_list= viewrecord.objects.all()
    return render (request,'base.html',{'viewrecord_list':viewrecord_list})


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
    # records = viewrecord.objects.all()  # Fetch records from the ViewRecord model
    license = request.GET.get('license')
    speed = request.GET.get('speed')
    date = request.GET.get('date')

    # Retrieve filtered records based on the search criteria
    filtered_records = viewrecord.objects.all()  # Fetch all records by default

    if license:
        filtered_records = filtered_records.filter(licenseplate_no__icontains=license)

    if speed:
        filtered_records = filtered_records.filter(speed__icontains=speed)
	
    if date:
        filtered_records = filtered_records.filter(date__icontains=date)

    # Create a response object with CSV content
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="view_records.csv"'

    # Create a CSV writer and write the header row
    writer = csv.writer(response)
    writer.writerow(['SN', 'License Plate No', 'Speed', 'Date', 'ID', 'Count'])

    # Write the data rows
    for record in filtered_records:
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

