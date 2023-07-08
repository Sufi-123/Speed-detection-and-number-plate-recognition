from django.db import models
from django.contrib.auth.models import AbstractUser


class viewrecord(models.Model):
    # user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)

    # SN= models.IntegerField(primary_key=True)
    liscenceplate_no= models.CharField(max_length=100)
    speed= models.IntegerField()
    # limit= models.IntegerField()
    # limit_crossed= models.IntegerField()
    date= models.DateField()
    IDs= models.IntegerField()
    count= models.IntegerField()
   


def __int__(self) :
    return int(self.SN)


class traffic(models.Model):
    TrafficBooth= models.IntegerField(primary_key=True)
    Areacode= models.IntegerField()
    location= models.CharField(max_length=100)
    
    
def __int__(self) :
    return int(self.Areacode)
   


    

