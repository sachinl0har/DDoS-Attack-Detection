# detection/models.py

from django.db import models

class TrainLog(models.Model):
    model = models.CharField(max_length=500)
    accuracy = models.CharField(max_length=500)

class TestLog(models.Model):
    model = models.CharField(max_length=500)
    accuracy = models.CharField(max_length=500)

class FileLog(models.Model):
    protocol_type = models.CharField(max_length=500)
    flag = models.CharField(max_length=500)
    service = models.CharField(max_length=500)
    is_ddos = models.CharField(max_length=500)
