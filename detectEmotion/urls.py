from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
     path('show_emotion/', views.show_emotion, name='show_emotion'),
]
