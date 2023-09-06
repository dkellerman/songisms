from django.urls import path
from . import views

urlpatterns = [
    path("", views.rhymes),
    path("completions/", views.completions),
    path("vote/", views.vote),
]
