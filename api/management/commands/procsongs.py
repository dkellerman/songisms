#!/usr/bin/env python

import multiprocessing as mp
from django.core.management.base import BaseCommand, CommandError
from api.models import *
from api.utils import fetch_audio, make_ngrams, set_lyrics_ipa, prune

class Command(BaseCommand):
    help = 'Process songs'

    def add_arguments(self, parser):
        parser.add_argument('--id', nargs='+', type=int)
        parser.add_argument('--force-update', '-u', action='store_true')
        parser.add_argument('--process', '-p', nargs='+', type=str, help='[all|rhymes|ngrams|audio|audiolink|ipa]')
        parser.add_argument('--dry-run', '-D', action='store_true')
        parser.add_argument('--no-prune', '-P', action='store_true')


    def handle(self, *args, **options):
        ids = options.get('id')
        dry_run = options.get('dry_run', False)
        no_prune = options.get('no_prune', False)
        force_update = options.get('force_update', False)

        process = options.get('process')
        if not process:
            raise CommandError('Please specify --process (-p) argument')

        process_rhymes = 'all' in process or 'rhymes' in process
        process_ngrams = 'all' in process or 'ngrams' in process
        process_audio = 'all' in process or 'audio' in process
        process_audiolinks = 'all' in process or 'audiolink' in process
        process_ipa = 'all' in process or 'ipa' in process

        if ids:
            songs = Song.objects.filter(pk__in=ids)
        else:
            songs = Song.objects.all()

        audio = []

        song_ct = songs.count()
        for idx, song in enumerate(songs):
            print(f'\n ===> {idx + 1} of {song_ct}', song.pk, song.title)

            if song.rhymes_raw:
                if process_rhymes and (force_update or not song.rhymes.count()):
                    print('\t[RHYMES]')
                    if not dry_run:
                        song.set_rhymes(song.rhymes_raw)

            if song.lyrics:
                if process_ngrams and (force_update or not song.ngrams.count()):
                    print('\t[NGRAMS]')
                    if not dry_run:
                        make_ngrams(song, force_update=force_update)

            if process_audiolinks:
                if (not song.youtube_id) and song.audio_file:
                    print('\t[UNLINKING AUDIO]', song.spotify_id)
                    if not dry_run:
                        song.audio_file = None
                        song.save()
                    return

                if song.audio_file_exists():
                    if not song.audio_file or force_update:
                        print('\t[LINK AUDIO]', song.spotify_id, song.audio_file_path)
                        if not dry_run:
                            song.audio_file = song.audio_file_path
                            song.save()
                else:
                    if song.audio_file:
                        print('\t[UNLINKING AUDIO]', song.spotify_id, song.audio_file_path)
                        if not dry_run:
                            song.audio_file = None
                            song.save()

            if song.youtube_id and process_audio:
                if force_update or not song.audio_file:
                    print('\t[AUDIO] Queuing')
                    if not dry_run:
                        audio.append(song)

            if song.lyrics:
                if process_ipa and (force_update or not song.lyrics_ipa):
                    print('\t[IPA]')
                    if not dry_run:
                        set_lyrics_ipa(song)

        if not no_prune:
            print('[PRUNING]')
            prune()

        if len(audio):
            if len(audio) > 1:
                print("Fetching audio", len(audio))
                mp.set_start_method('fork')
                with mp.Pool(mp.cpu_count()) as p:
                    p.map(fetch_audio_wrapper, audio)
            else:
                fetch_audio(audio[0], convert=False)


def fetch_audio_wrapper(song):
    try:
        return fetch_audio(song, convert=False)
    except:
        print("Error fetching audio", song.pk, song.title)
