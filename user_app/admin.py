from django.contrib import admin
from user_app.models import viewrecord,Person,vehicle

# Register your models here.
admin.site.register(viewrecord)
admin.site.register(Person)
admin.site.register(vehicle)