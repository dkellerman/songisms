#!/usr/bin/env python

import multiprocessing as mp
from django.core.management.base import BaseCommand
from api.models import *
from api.utils.cloud import fetch_stems


class Command(BaseCommand):
    help = 'Process stems'

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        songs = Song.objects.all()
        queue = []
        song_ct = songs.count()

        for idx, song in enumerate(songs):
            print(f'\n ===> {idx + 1} of {song_ct}', song.pk, song.title)

            if len(song.attachments.filter(attachment_type='vocals')) == 0 and song.audio_file:
                queue.append(song)

        if len(queue):
            print("Fetching stems", len(queue))
            mp.set_start_method('fork')
            with mp.Pool(mp.cpu_count()) as p:
                p.map(fetch_wrapper, queue[0:100])


def fetch_wrapper(song):
    try:
        return fetch_stems(song)
    except:
        print("Error fetching stems", song.pk, song.title)
