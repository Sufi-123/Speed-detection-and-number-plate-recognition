# from django.db import models
# from django.contrib.auth.models import AbstractUser


# class Record(models.Model):
#     # user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)

#     # SN= models.IntegerField(primary_key=True)
#     liscenseplate_no= models.CharField(max_length=100)
#     speed= models.IntegerField()
#     # limit= models.IntegerField()
#     # limit_crossed= models.IntegerField()
#     date= models.DateField()
#     IDs= models.IntegerField()
#     count= models.IntegerField()

#     def __str__(self):
#         return self.date.strftime("%d/%m/%Y")
   


# def __int__(self) :
#     return int(self.SN)


# class traffic(models.Model):
#     TrafficBooth= models.IntegerField(primary_key=True)
#     Areacode= models.IntegerField()
#     location= models.CharField(max_length=100)
    
    
# def __int__(self) :
#     return int(self.Areacode)

from django.db import models

# Create your models here.




class Record(models.Model):
    
    speed= models.IntegerField()
    date= models.DateField()
    count= models.IntegerField()
    liscenseplate_no= models.CharField(max_length=50)

   



    

