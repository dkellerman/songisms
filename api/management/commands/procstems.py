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
            print(f'\n ===> {idx + 1} of {song_ct}', song.pk, song.title, song.audio_file)
            for a in song.attachments.all():
                print('\t', a.attachment_type, a.file)

            if len(song.attachments.filter(attachment_type='vocals')) == 0 and song.audio_file:
                queue.append(song)

        if len(queue):
            print("Fetching stems", len(queue))
            mp.set_start_method('fork')
            with mp.Pool(mp.cpu_count()) as p:
                p.map(fetch_wrapper, queue[10:100])


def fetch_wrapper(song):
    try:
        print("==> Fetching stems", song.pk, song.title)
        return fetch_stems(song)
    except Exception as err:
        print("Error fetching stems", err, song.pk, song.title)
