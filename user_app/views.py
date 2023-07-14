# from speed_estimation.vehicle_speed_count import process_video
from speed_estimation.combined import process_video
from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse
from .models import Record
import csv
from django.shortcuts import render
from django.contrib.auth import authenticate, login
import re,uuid

#view for welcome page and mac-address authorization
def welcome_page(request):
    #checking if the user is already authorized:
    try:
        if request.user.is_authenticated:
            return render(request, 'base.html')
        if request.method == 'POST':
            mac_address = (':'.join(re.findall('..', '%012x' % uuid.getnode())))
            user = authenticate(request,mac_address=mac_address)
            if user is not None:
                login(request, user)
                # Redirect to the appropriate page
                return render(request,'base.html')
                    
            else:
                # Handle invalid login
                return render(request, 'welcome_dashboard.html', {'error': 'Invalid MAC address'})
            
    #throw exception for user authentication        
    except Exception as e:
        print(e)

    return render(request, 'welcome_dashboard.html')

# Create your views here.
def home (request):
    try:
        if request.user.is_authenticated:
            Record_list= Record.objects.all()
        return render (request,'base.html',{'Record_list':Record_list})

    except Exception as e:
        print(e)     
    
#authorize mac address:

def login_view(request):

        if request.method == 'POST':
            mac_address = (':'.join(re.findall('..', '%012x' % uuid.getnode())))
            user = authenticate(request,mac_address=mac_address)
            if user is not None:
                login(request, user)
                # Redirect to the appropriate page
                return render(request,'base.html')
            else:
                # Handle invalid login
                return render(request, 'welcome_dashboard.html', {'error': 'Invalid MAC address'})

        return render(request, 'welcome_dashboard.html')


def video(request):
    return StreamingHttpResponse(process_video(), content_type='multipart/x-mixed-replace; boundary=frame')

def Records(request):
    Record_list= Record.objects.all()
    context = {
        'Record_list': Record_list
    }
    return render(request, 'Records.html', context)

def download_csv(request):
    # Retrieve data from the database or any other source
    # records = Record.objects.all()  # Fetch records from the ViewRecord model
    license = request.GET.get('license')
    speed = request.GET.get('speed')
    date = request.GET.get('date')

    # Retrieve filtered records based on the search criteria
    filtered_records = Record.objects.all()  # Fetch all records by default

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
    writer.writerow(['SN', 'StationID', 'License Plate No', 'Speed', 'Date', 'ID', 'Count'])

    # Write the data rows
    for record in filtered_records:
        writer.writerow([
            record.pk,
            record.stationID,
            record.liscenseplate_no,
            record.speed,
            record.date,
            record.count
        ])

    return response

