#!/usr/bin/env python

import argparse
from django.core.management.base import BaseCommand
from api.models import *


class Command(BaseCommand):
    help = 'Check stuff'

    def add_arguments(self, parser):
        parser.add_argument('--audio', '-a', action=argparse.BooleanOptionalAction)
        parser.add_argument('--stems', '-s', action=argparse.BooleanOptionalAction)
        parser.add_argument('--new', '-n', action=argparse.BooleanOptionalAction)
        parser.add_argument('--lyrics', '-l', action=argparse.BooleanOptionalAction)
        parser.add_argument('--writers', '-w', action=argparse.BooleanOptionalAction)
        parser.add_argument('--rhymes', '-r', action=argparse.BooleanOptionalAction)
        parser.add_argument('--style', '-S', action=argparse.BooleanOptionalAction)
        parser.add_argument('--metadata', '-m', action=argparse.BooleanOptionalAction)
        parser.add_argument('--transcripts', '-t', action=argparse.BooleanOptionalAction)

    def handle(self, *args, **options):
        songs = Song.objects.all()
        check_new, check_audio, check_stems, check_lyrics, check_writers, check_rhymes, \
        check_style, check_metadata, check_transcript = \
            [options[k] for k in ('new', 'audio', 'stems', 'lyrics', 'writers', \
                                  'rhymes', 'style', 'metadata', 'transcripts')]

        for idx, song in enumerate(songs):
            if check_transcript:
                l = song.attachments.filter(attachment_type='transcript_line').exists()
                w = song.attachments.filter(attachment_type='transcript_word').exists()
                if not l and not w:
                    print("[NO TRANSCRIPTS]", song.spotify_id, song.title)
                else:
                    if not l:
                        print("[NO TRANSCRIPT BY LINE]", song.spotify_id, song.title)
                    if not w:
                        print("[NO TRANSCRIPT BY WORD]", song.spotify_id, song.title)
            if check_metadata:
                if not song.metadata:
                    print("[NO METADATA]", song.spotify_id, song.title)
                else:
                    if not song.metadata.get('spotify'):
                        print("[NO SPOTIFY METADATA]", song.spotify_id, song.title)
                    if not song.metadata.get('songMeanings'):
                        print("[NO SONGMEANINGS METADATA]", song.spotify_id, song.title)
                    if not song.metadata.get('youtube'):
                        print("[NO YOUTUBE METADATA]", song.spotify_id, song.title)
                    if not song.metadata.get('ipa'):
                        print("[NO IPA METADATA]", song.spotify_id, song.title)
            if check_rhymes:
                if not song.rhymes_raw.strip():
                    print("[NO RHYMES]", song.spotify_id, song.title)
            if check_lyrics or check_style:
                if not song.lyrics.strip():
                    print("[NO LYRICS]", song.spotify_id, song.title)
                if check_style:
                    if "\n\n" not in song.lyrics:
                        print("[STYLE: NO SECTIONS?]", song.spotify_id, song.title)
                    if "." in song.lyrics or "..." in song.lyrics:
                        print("[STYLE: BAD PUNCTUATION?]", song.spotify_id, song.title)
                    if song.lyrics.count("-") > 3:
                        print("[STYLE: DASHED VOX?]", song.spotify_id, song.title)
            if check_writers:
                if not song.writers.exists():
                    print("[NO WRITERS]", song.spotify_id, song.title)
            if check_stems:
                if not song.attachments.filter(attachment_type='vocals').exists():
                    print("[NO VOX]", song.spotify_id, song.title)
            if check_new and song.is_new:
                print('\t[NEW]', song.spotify_id, song.title)
            if check_audio:
                if song.youtube_id:
                    if song.audio_file_exists():
                        if not song.audio_file:
                            print('\t[UNLINKED AUDIO]', song.spotify_id)
                        elif song.audio_file.size < 100 * 1024:
                            print('\t[EMPTY AUDIO?]', song.spotify_id)
                    else:
                        if not song.audio_file:
                            print('\t[UNDOWNLOADED AUDIO]', song.spotify_id)
                else:
                    if song.audio_file:
                        print('\t[INVALID AUDIO]', song.spotify_id)
                    else:
                        print('\t[NO YOUTUBE ID]', song.spotify_id)
