#!/usr/bin/env python

import multiprocessing as mp
from django.core.management.base import BaseCommand
from api.models import *
from api.utils.cloud import fetch_stems


class Command(BaseCommand):
    help = 'Process stems'

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
            for a in song.attachments.all():
                print('\t', a.attachment_type, a.file)

            if len(song.attachments.filter(attachment_type='vocals')) == 0 and song.audio_file:
                queue.append(song)

        print("Queue size (TOTAL):", len(queue))
        if len(queue):
            print("Fetching stems", min(len(queue), limit or 0))
            mp.set_start_method('fork')
            with mp.Pool(mp.cpu_count()) as p:
                q = queue[0:limit] if limit else queue
                p.map(fetch_wrapper, q)


def fetch_wrapper(song):
    try:
        print("==> Fetching stems", song.pk, song.title)
        return fetch_stems(song)
    except Exception as err:
        print("Error fetching stems", err, song.pk, song.title)
