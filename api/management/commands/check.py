#!/usr/bin/env python

import argparse
from django.core.management.base import BaseCommand
from api.models import *


class Command(BaseCommand):
    help = 'Check stuff'

    def add_arguments(self, parser):
        parser.add_argument('--audio', '-a', action=argparse.BooleanOptionalAction)

    def handle(self, *args, **options):
        songs = Song.objects.all()
        song_ct = songs.count()
        audio = options['audio']

        for idx, song in enumerate(songs):
            # print(f'\n ===> {idx + 1} of {song_ct}', song.pk, song.spotify_id, song.title)
            if song.youtube_id:
                if song.audio_file_exists():
                    if not song.audio_file:
                        print('\t[UNLINKED AUDIO]', song.spotify_id)
                else:
                    if not song.audio_file:
                        print('\t[UNDOWNLOADED AUDIO]', song.spotify_id)
            else:
                if song.audio_file:
                    print('\t[INVALID AUDIO]', song.spotify_id)
                else:
                    print('\t[NO YOUTUBE ID]', song.spotify_id)
