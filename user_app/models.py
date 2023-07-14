from django.db import models


# Create your models here.


class Station(models.Model):
    IDs= models.AutoField(primary_key=True)
    location = models.CharField(max_length=80)
    areacode = models.PositiveIntegerField()
    mac_address = models.CharField(max_length=17)
    
class Record(models.Model):
    stationID= models.ForeignKey('Station', on_delete=models.CASCADE)
    speed= models.IntegerField()
    date= models.DateField()
    count= models.IntegerField()
    liscenseplate_no= models.CharField(max_length=50, null=True)

   



    

