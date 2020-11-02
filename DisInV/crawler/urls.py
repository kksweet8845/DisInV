from django.urls import path

from .views import test_pts_crawler, test_cts_crawler, test_ltn_crawler, test_nowNews_crawler

urlpatterns = [
    path('api/pts/', test_pts_crawler, name='test pts'),
    path('api/cts/', test_cts_crawler, name='test cts'),
    path('api/ltn/', test_ltn_crawler, name='test ltn'),
    path('api/nowNews/', test_nowNews_crawler, name='test now News')
]