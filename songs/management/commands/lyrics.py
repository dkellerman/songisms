#!/usr/bin/env python ./manage.py script

import json
import argparse
import os
import re
from django.core.management.base import BaseCommand
from songs.models import Song
from syrics.api import Spotify
from songisms import utils

# SP_DC = "AQAED3nImGLfmFmyR4Aujm_VirXWtQn8QZcsGmK8wHfFnDtpxgFnInZRj4pVpvmA59kmS_pL0-E79Akb_dRR5sfVQgPehEH_O5mybSbFct2GZQV6iS-U6LR7xp2XIdnLQQluR1GtNnb3COhwTj1Yj97CsSlJa1QY"

class Command(BaseCommand):
    help = '''Fetch LRC lyrics from spotify, only pre-fill lyrics if not already set
              (even when using --force flag)'''

    def add_arguments(self, parser):
        parser.add_argument('--id', '-i', type=str, default=None)
        parser.add_argument('--limit', '-l', type=int, default=0)
        parser.add_argument('--force', '-F', action=argparse.BooleanOptionalAction)
        parser.add_argument('--auth', '-a', type=str, default='')

    def handle(self, *args, **options):
        if not options['auth']:
            print("Please provide a Spotify auth code (sp_dc) with --auth")
            return
        sp = Spotify(options['auth'])

        songs = Song.objects.all()
        if options['id']:
            songs = songs.filter(spotify_id__in=options['id'].split(','))
        limit = options['limit']
        queue = []

        for s in songs:
            if not s.metadata or not s.metadata.get('lrc') or options['force']:
                print('queue', s.title, s.spotify_id)
                queue.append(s)

        print("Queue size (TOTAL):", len(queue))
        if len(queue):
            print("Processing", min(len(queue), limit or len(queue)))
            for song in queue[0:limit] if limit else queue:
                wrapper(song, sp)


def wrapper(song, sp):
    path = f"./data/spotify/{song.spotify_id}.json"
    if not os.path.exists(path):
        print('=> fetching lrc...')
        lrc = sp.get_lyrics(song.spotify_id)
        with open(path, "w") as f:
            f.write(json.dumps(lrc, indent=2))
    else:
        with open(path, "r") as f:
            lrc = json.loads(f.read())
    if lrc is None:
        print("NO LRC FOUND", song.spotify_id, song.title)
        return
    song.metadata = song.metadata or {}
    song.metadata['lrc'] = lrc

    if not song.lyrics:
        print('=> adding lyrics...')
        lyr = []
        for obj in lrc['lyrics']['lines']:
            line = utils.normalize_lyric(obj['words'])
            line = re.sub(r'♪', '\n', line)
            line = re.sub(r'\bi\b', 'I', line)
            lyr.append(line)
        song.lyrics = '\n'.join(lyr).strip()

    song.save()


