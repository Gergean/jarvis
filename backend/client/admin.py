from django.contrib import admin
from client.models import Client


class ClientAdmin(admin.ModelAdmin):
    search_fields = ["id", "username", "firstname", "lastname"]
    list_display = ["id", "username", "firstname", "lastname", "deposited_amount"]


admin.site.register(Client, ClientAdmin)