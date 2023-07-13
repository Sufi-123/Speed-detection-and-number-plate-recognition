from django.db import models

# Create your models here.
class Station(models.Model):
    # stationID= models.ForeignKey('Station', on_delete=models.CASCADE)
    IDs= models.IntegerField()
    location = models.CharField(max_length=80)
    areacode = models.PositiveIntegerField()
    mac_address = models.CharField(max_length=17)