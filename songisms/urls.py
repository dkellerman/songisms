from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('songs/', include('songs.urls')),
    path('rhymes/', include('rhymes.urls')),
    path('admin/', include('smuggler.urls')),
    path('admin/', admin.site.urls),
    path('__debug__/', include('debug_toolbar.urls')),
]
