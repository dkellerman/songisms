#!/usr/bin/env python

import multiprocessing as mp
from django.core.management.base import BaseCommand
from api.models import *
from api.utils.gpt import gpt_fetch_ipa


class Command(BaseCommand):
    help = 'Process IPA'

    def add_arguments(self, parser):
        parser.add_argument('--id', '-i', type=str, default=None)
        parser.add_argument('--limit', '-l', type=int, default=10)


    def handle(self, *args, **options):
        songs = Song.objects.all()
        if options['id']:
            songs = songs.filter(spotify_id__in=options['id'].split(','))
        limit = options['limit']
        queue = []

        for idx, song in enumerate(songs):
            if not song.metadata or not song.metadata.get('ipa'):
                queue.append(song)

        print("Queue size (TOTAL):", len(queue))
        if len(queue):
            print("Fetching", min(len(queue), limit or 0))
            mp.set_start_method('fork')
            with mp.Pool(mp.cpu_count()) as p:
                q = queue[0:limit] if limit else queue
                p.map(fetch_wrapper, q)


def fetch_wrapper(song):
    try:
        print("==> Fetching IPA", song.pk, song.title)
        return gpt_fetch_ipa(song.lyrics)
    except Exception as err:
        print("Error fetching IPA", err, song.pk, song.title)

