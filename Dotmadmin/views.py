# Create your views here.
from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import redirect, render
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from user_app.visualize import generate_bar_graph, generate_line_graph,generate_permonth_graph,generate_perday_graph
from collections import defaultdict

from django.db import connection

from user_app.models import Record, Station

from .forms import StationForm

#views for admin login authentication
def admin_login(request):
    try:
        if request.user.is_authenticated:
            return render(request, 'dotm_home.html')

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


# view for regrtation of new station
def register_station(request):
    if request.method == 'POST':
        print('success')
        form = StationForm(request.POST)
        print('success')
        if form.is_valid():
            print('success')
            form.save()
            return JsonResponse({'message': 'Station registered successfully!'})
        else:
            return JsonResponse({'error': 'Invalid form data.'})
            # Redirect to success page or do something else
            # message = {'message': 'Station registered successfully!'}
            # return render(request,'station_list.html',message)

    else:
        form = StationForm()
    
    context = {'form': form}
    return render(request, 'station_list.html', context)

def traffics(request):
    Station_list= Station.objects.all()
    
    return render(request, 'station_list.html',
		  {'Station_list':Station_list}) 
            

 



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

    context={
        'chart_path': chart_path,
        'graph_path': graph_path,
        'permonth_path' : permonth_path,
        'perday_path' : perday_path
          

    }
    return render(request, 'chart.html',context)

def notice(request):
    return render ( request ,'notice.html',)

def welcome_view(request):
	return render(request,'welcome_dashboard.html')

def dotm(request):
	return render(request,'dotm_dashboard.html')

def dotm_home(request):
	return render(request,'dotm_home.html')


