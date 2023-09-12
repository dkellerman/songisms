import json
from django import http
from django.views.decorators.http import require_http_methods
from django.shortcuts import render
from django.forms.models import model_to_dict
from .models import Song


@require_http_methods(["GET"])
def home(request):
    q = request.GET.get("q", "")
    hits = []
    for s in Song.objects.query(q or None):
        hit = model_to_dict(s, fields=['id', 'title', 'spotify_id', 'artists'])
        hits.append(hit)
    print(hits)
    context = dict(hits=hits, q=q)
    return render(request, "songs.html", context=context)
