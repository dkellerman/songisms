import argparse
from django.core.management.base import BaseCommand
from songs.models import Song
from songisms import utils


class Command(BaseCommand):
    help = 'Process IPA'

    def add_arguments(self, parser):
        parser.add_argument('--id', '-i', type=str, default=None)
        parser.add_argument('--limit', '-l', type=int, default=10)
        parser.add_argument('--force-fetch', '-F', action=argparse.BooleanOptionalAction)


    def handle(self, *args, **options):
        songs = Song.objects.filter(is_new=False).exclude(lyrics=None)
        if options['id']:
            songs = songs.filter(spotify_id__in=options['id'].split(','))
        limit = options['limit']
        queue = []

        for song in songs:
            if not song.metadata or not song.metadata.get('ipa') or options['force_fetch']:
                queue.append(song)

        print("Queue size (TOTAL):", len(queue))
        if len(queue):
            print("Fetching", min(len(queue), limit or 0))
            for song in queue[0:limit] if limit else queue:
                fetch_wrapper(song)


def fetch_wrapper(song):
    try:
        print("==> Fetching IPA", song.pk, song.title)
        ipa = utils.gpt_fetch_ipa(song.lyrics)
        song.metadata = song.metadata or {}
        song.metadata['ipa'] = ipa
        song.save()
    except Exception as err:
        print("Error fetching IPA", err, song.pk, song.title)

