#!/usr/bin/env python

import multiprocessing as mp
from django.core.management.base import BaseCommand
from api.models import *
from api.utils import fetch_audio


class Command(BaseCommand):
    help = 'Process audio'

    def add_arguments(self, parser):
        parser.add_argument('--id', '-i', type=str, default=None)

    def handle(self, *args, **options):
        songs = Song.objects.all()
        if options['id']:
            songs = songs.filter(spotify_id__in=options['id'].split(','))

        audio_queue = []

        for idx, song in enumerate(songs):
            if song.youtube_id and not song.audio_file:
                print('\t[QUEUEING AUDIO]', song.pk, song.spotify_id, song.audio_file_path)
                audio_queue.append(song)

        print("=> Queued:", len(audio_queue))
        if len(audio_queue):
            if len(audio_queue) > 1:
                print("Fetching audio", len(audio_queue))
                mp.set_start_method('fork')
                with mp.Pool(mp.cpu_count()) as p:
                    p.map(fetch_audio_wrapper, audio_queue)
            else:
                fetch_audio(audio_queue[0], convert=False)


def fetch_audio_wrapper(song):
    try:
        return fetch_audio(song, convert=False)
    except:
        print("Error fetching audio", song.pk, song.title)
