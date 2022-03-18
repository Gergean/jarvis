from django.db import models


class Client(models.Model):
    id = models.IntegerField(primary_key=True)
    username = models.CharField(max_length=100, null=True, blank=True)
    firstname = models.CharField(max_length=100, null=True, blank=True)
    lastname = models.CharField(max_length=100, null=True, blank=True)
    deposited_amount = models.DecimalField(
        default=0, decimal_places=8, max_digits=30)
    binance_api_key = models.CharField(max_length=64, null=True, blank=True)
    binance_api_secret_key = models.CharField(max_length=64, null=True, blank=True)
    withdrawal_address = models.CharField(max_length=64, null=True, blank=True)

    def __str__(self):
        return f"Client #{self.id}"