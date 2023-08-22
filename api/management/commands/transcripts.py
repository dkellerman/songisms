#!/usr/bin/env python

import argparse
import json
import multiprocessing as mp
from django.core.management.base import BaseCommand
from api.models import *
from api.utils.cloud import fetch_workflow


class Command(BaseCommand):
    help = 'Process transcripts'

    def add_arguments(self, parser):
        parser.add_argument('--id', '-i', type=str, default=None)
        parser.add_argument('--limit', '-l', type=int, default=10)
        parser.add_argument('--force-fetch', '-F', action=argparse.BooleanOptionalAction)
        parser.add_argument('--force-process', '-P', action=argparse.BooleanOptionalAction)

    def handle(self, *args, **options):
        songs = Song.objects.all()
        if options['id']:
            songs = songs.filter(spotify_id__in=options['id'].split(','))
        limit = options['limit']
        force_fetch = options['force_fetch']
        force_process = options['force_process']
        queue = []

        for idx, song in enumerate(songs):
            if song.audio_file and (not song.get_attachment('transcript_word') or force_fetch):
                queue.append(song)
            elif force_process and song.has_attachment('transcript_word'):
                process_transcript(song)

        print("Queue size (TOTAL):", len(queue))
        if len(queue):
            print("Fetching", min(len(queue), limit or 0))
            mp.set_start_method('fork')
            with mp.Pool(mp.cpu_count()) as pool:
                q = queue[0:limit] if limit else queue
                pool.map(fetch_wrapper, q)


def fetch_wrapper(song):
    try:
        print("==> Fetching", song.pk, song.title)
        fetch_workflow('transcript', song)
        process_transcript(song)
    except Exception as err:
        print("Error fetching transcript", err, song.pk, song.title)


def process_transcript(song):
    print("==> Processing", song.pk, song.title)
    a = song.get_attachment('transcript_word')
    if not a or not a.file:
        print("ERROR: No transcript found for", song.pk, song.title)
        return
    transcript = json.loads(a.file.read())
    song.metadata = song.metadata or {}
    song.metadata['transcript'] = transcript
    song.save()
