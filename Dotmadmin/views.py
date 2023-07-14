# Create your views here.

from user_app.models import Record, Station
from django.shortcuts import redirect, render
from django.contrib import messages
from django.contrib.auth import authenticate, login
from user_app.visualize import *
from collections import defaultdict


from django.db import connection




#views for admin login authentication
def admin_login(request):
    try:
        if request.user.is_authenticated:
            return redirect( 'Home')

        if request.method == 'POST':
            username = request.POST.get('username')
            password = request.POST.get('password')

            user = authenticate(request, username=username, password=password)

            if user is not None and user.is_superuser:
                login(request, user)
                return render(request, 'dotm_home.html')
            else:
                messages.error(request, 'Invalid username or password')
                return redirect('dotm_login')

        return render(request, 'dotm_login.html')

    except Exception as e:
        print(e)


#views for dotm homepage
def dotm_home(request):
        #passing 4 records from station
        station=Station.objects.all()[:4]
        context={
            'stations': station
        }
        return render(request,'dotm_home.html', context)


def traffics(request):
    Station_list= Station.objects.all()
    if request.method == 'POST':

        stationID = request.POST['stationID']
        Areacode = request.POST['areaCode']
        location = request.POST['location']
        mac_address = request.POST['mac_address']


        station_booth = Station.objects.create(IDs=stationID, areacode=Areacode, location=location, mac_address=mac_address)
        station_booth.save()


        return redirect('traffics')  # Replace 'traffic_list' with the URL name of your traffic list view

    return render(request, 'station_list.html',
          {'Station_list':Station_list})  # Replace 'your_template.html' with the actual template file name

# def traffics(request):
#     if request.method == 'POST':
#         form = StationForm(request.POST)
#         if form.is_valid():
#             form.save()
#             print('sufs')
#             return redirect('traffics')  # Redirect to success page after form submission
#     else:
#         form = StationForm()

#     return render(request, 'station_list.html', {'form': form})
            

def chart_display(request):
    limit = 50  # Speed limit value
    records = Record.objects.order_by('speed')[:20][::-1]  # Get the bottom 20 records by speed
    new_record= Record.objects.order_by('speed')
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

    # chart_path,graph_path,permonth_path,perday_path=visualize_charts()
    context={
        'chart_path': chart_path,
        'graph_path': graph_path,
        'permonth_path' : permonth_path,
        'perday_path' : perday_path
 

    }
    return render(request, 'chart.html',context)

def notice(request):
    return render ( request ,'notice.html',)




