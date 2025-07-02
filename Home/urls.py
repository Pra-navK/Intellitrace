from django.urls import path
from .views import index, run_tracer_view

urlpatterns = [
    path('', index),              # Home page
    path('api/trace/', run_tracer_view),  # POST endpoint
]
