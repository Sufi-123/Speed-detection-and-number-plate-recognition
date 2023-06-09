from django.shortcuts import render
from django.http import HttpResponse
from .models import Person , viewrecord

# Create your views here.
def home (request):
    return render (request,'base.html')

# def viewrecords (request):
#     return render (request,'viewrecords.html')


from django.shortcuts import render
from django.db import connection

def viewrecords(request):
    viewrecord_list= viewrecord.objects.all()
    return render ( request ,'viewrecords.html',
                   {'viewrecord_list': viewrecord_list})
# def viewrecords(request):

#     # with connection.cursor() as cursor:
#     #     cursor.execute("SELECT * FROM viewrecordss")
#     #     rows = cursor.fetchall()
#     # return render(request, 'viewrecords.html', {'rows': rows})

