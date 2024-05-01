# intrusion_detection_backend/urls.py

from django.urls import path
from detection import views

urlpatterns = [
    path('api/train/', views.train_model),
    path('api/test/', views.test_model),
    path('api/reports/', views.reports_view, name='reports'),
]
