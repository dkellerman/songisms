#!/usr/bin/env python

import argparse
import json
import multiprocessing as mp
from django.core.management.base import BaseCommand
from api.models import Song
from api.utils.cloud import fetch_workflow
from api.utils.text2 import *


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

        for song in songs:
            if (not force_process and song.audio_file and
               (not song.get_attachment('transcript_word') or force_fetch)
            ):
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


class TSWord:
    def __init__(self, data):
        self.data = data
    def __str__(self):
        return self.data['text']
    def __eq__(self, val):
        return self.data['text'] == val


def process_transcript(song):
    print("==> Processing", song.pk, song.title)

    a = song.get_attachment('transcript_word')
    if not a or not a.file:
        print("ERROR: No transcript found for", song.pk, song.title)
        return

    transcript = json.loads(a.file.read())
    tr_words = [TSWord(data) for data in transcript if data['text'] not in ['<EOL>', '<SOL>']]
    for obj in tr_words:
        obj.data['text'] = normalize_lyric(obj.data['text'])
    song_words = sum([ tokenize_lyric(l) for l in song.lyrics.split('\n') if l.strip() ], [])

    aligned_song_words, aligned_tr_words, _, _ = align_vals(song_words, tr_words)

    slots = []
    slot = None
    last_slot = None

    for word, tr_obj in zip(aligned_song_words, aligned_tr_words):
        word = str(word)
        has_word = word != '_'
        tr_word = str(tr_obj)
        has_tr = tr_word != '_' and tr_word == word

        if not has_word:
            continue

        slot = slot or dict()
        slot['text'] = slot.get('text', '') + word + ' '
        if not has_tr:
            slot['start'] = slot.get('start',
                                     last_slot['end'] + .001 if last_slot
                                     else transcript[0]['start'])
        else:
            slot['text'] = slot['text'].strip()
            slot['start'] = slot.get('start', tr_obj.data['start'])
            slot['end'] = tr_obj.data['end']

            # print('->', slot['text'], f"{slot['start']}-{slot['end']} ({slot['end'] - slot['start']})")
            slots.append(slot)
            last_slot = slot
            slot = None

    song.metadata = song.metadata or {}
    song.metadata['transcript'] = slots
    with open(f"./data/transcripts/{song.title}.json", 'w') as f:
        f.write(json.dumps(slots, indent=2, ensure_ascii=False))
    song.save()
