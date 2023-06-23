from django.db import models


# Create your models here.
class Person (models.Model):
    full_name= models.CharField(max_length=100)
    def __str__(self) :
        return self.full_name
    
class viewrecord(models.Model):
    SN= models.IntegerField(primary_key=True)
    liscenceplate_no= models.IntegerField()
    speed= models.IntegerField()
    # limit= models.IntegerField()
    # limit_crossed= models.IntegerField()
    date= models.DateField()


def __int__(self) :
    return int(self.SN)

class vehicle(models.Model):
     ID= models.IntegerField()
     count= models.IntegerField()
    
def __int__(self) :
    return int(self.ID)
    

