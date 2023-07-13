from django import forms
from Dotmadmin.models import Station
import re


class StationForm(forms.ModelForm):
    class Meta:
        model = Station
        fields = ['IDs', 'areacode', 'location', 'mac_address']

    def clean_location(self):
        location = self.cleaned_data['location']
        # Add your validation logic for the location field
        if len(location) < 3:
            print("Location validation error: Location must have at least 3 characters.")
            raise forms.ValidationError("Location must have at least 3 characters.")
        return location
    
    def clean_areacode(self):
        areacode = self.cleaned_data['areacode']
        # Add your validation logic for the areacode field
        if areacode < 1000 or areacode > 9999:
            print("Areacode must be a 4-digit number.")
            raise forms.ValidationError("Areacode must be a 4-digit number.")
        return areacode
    
    # def clean_mac_address(self):
    #     mac_address = self.cleaned_data['mac_address']
    #     pattern = r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'
    #     if re.match(pattern, mac_address):
    #         print("Incorrect address pattern")
    #         raise forms.ValidationError("Incorrect address pattern")
    #     else:
    #         return mac_address
