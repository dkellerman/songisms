from django.contrib import admin
from django.urls import path, include
from django.apps import apps

urlpatterns = []

if apps.is_installed('songs'):
    urlpatterns += [path('songs/', include('songs.urls'))]
if apps.is_installed('rhymes'):
    urlpatterns += [path('rhymes/', include('rhymes.urls'))]

urlpatterns += [
    path('admin/', include('smuggler.urls')),
    path('admin/', admin.site.urls),
    path('__debug__/', include('debug_toolbar.urls')),
]
