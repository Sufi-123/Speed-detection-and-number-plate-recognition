from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model
from .models import Station

class MACAddressAuthBackend(ModelBackend):
    def authenticate(self, request, mac_address=None):
        User = get_user_model()
        try:
            station_user = Station.objects.get(mac_address=mac_address)
            user, _ = User.objects.get_or_create(username=f"station_{station_user.pk}")
            return user
        except Station.DoesNotExist:
            return None

# from django.contrib.auth.backends import ModelBackend
# from django.contrib.auth import get_user_model
# from .models import User, Station


# class MultiLoginBackend(ModelBackend):
#     def authenticate(self, request, username=None, password=None, mac_address=None, **kwargs):
#         User = get_user_model()

#         if username is not None and password is not None:
#             # Authenticate Dotm user with username and password
#             try:
#                 dotm_user = User.objects.get(username=username, is_DoTM=True)
#                 if dotm_user.check_password(password):
#                     return dotm_user
#             except User.DoesNotExist:
#                 return None

#         if mac_address is not None:
#             # Authenticate Station user with MAC address
#             try:
#                 station = Station.objects.get(mac_address=mac_address)
#                 return station.user
#             except Station.DoesNotExist:
#                 return None

#         return None

#     def get_user(self, user_id):
#         User = get_user_model()

#         try:
#             return User.objects.get(pk=user_id)
#         except User.DoesNotExist:
#             return None
