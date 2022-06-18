#!/usr/bin/env python

import multiprocessing as mp
from django.core.management.base import BaseCommand
from api.models import *
from api.cloud_utils import fetch_audio


class Command(BaseCommand):
    help = 'Process audio'

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        songs = Song.objects.all()
        audio_queue = []
        song_ct = songs.count()

        for idx, song in enumerate(songs):
            print(f'\n ===> {idx + 1} of {song_ct}', song.pk, song.title)

            if song.youtube_id:
                if song.audio_file_exists():
                    if not song.audio_file:
                        print('\t[LINK AUDIO]', song.spotify_id, song.audio_file_path)
                        song.audio_file = song.audio_file_path
                        song.save()
                    else:
                        if song.audio_file:
                            print('\t[UNLINKING AUDIO]', song.spotify_id)
                            song.audio_file = None
                            song.save()
                else:
                    if song.audio_file:
                        print('\t[AUDIO] Queuing')
                        audio_queue.append(song)
            else:
                if song.audio_file:
                    print('\t[UNLINKING AUDIO]', song.spotify_id, song.audio_file_path)
                    song.audio_file = None
                    song.save()

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
