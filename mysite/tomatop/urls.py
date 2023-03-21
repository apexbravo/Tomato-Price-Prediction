from django.urls import path
from .views import predict, predict_price
from . import views

app_name = 'tomatop'
urlpatterns = [
    path('', views.index, name='index'),
    path('predict_price/', predict_price, name='predict_price'),
    path('predict/', predict, name='predict'),
    path('predictImage/', views.predictImage, name='predictImage'),
    path('predictPrice/', views.predictPrice, name='predictPrice')
]
