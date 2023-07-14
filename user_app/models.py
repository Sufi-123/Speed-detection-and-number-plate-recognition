from django.db import models
from django.contrib.auth.models import User

# Create your models here.


class Station(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    IDs= models.IntegerField()
    location = models.CharField(max_length=80)
    areacode = models.PositiveIntegerField()
    mac_address = models.CharField(max_length=17)
    
class Record(models.Model):
    stationID= models.ForeignKey('Station', on_delete=models.CASCADE)
    speed= models.IntegerField()
    date= models.DateField()
    count= models.IntegerField()
    liscenseplate_no= models.CharField(max_length=50, null=True)

   



    

