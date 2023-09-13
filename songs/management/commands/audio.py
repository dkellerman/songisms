import argparse
from django.core.management.base import BaseCommand
from songs.models import Song
from songisms import utils


class Command(BaseCommand):
    help = 'Fetch and process song audio'

    def add_arguments(self, parser):
        parser.add_argument('--id', '-i', type=str, default=None)
        parser.add_argument('--fetch', '-f', action=argparse.BooleanOptionalAction)
        parser.add_argument('--force-fetch', '-F', action=argparse.BooleanOptionalAction)
        parser.add_argument('--prune', '-P', action=argparse.BooleanOptionalAction)
        parser.add_argument('--limit', '-l', type=int, default=None)

    def handle(self, *args, **options):
        if options['prune']:
            self.prune()
            return
        elif not options['fetch'] and not options['force_fetch']:
            print("specify --fetch or --force-fetch")
            return

        songs = Song.objects.exclude(youtube_id=None)
        if options['id']:
            songs = songs.filter(spotify_id__in=options['id'].split(','))

        audio_queue = []

        for song in songs:
            if not song.audio_file_exists() or options['force_fetch']:
                audio_queue.append(song)

        if options['limit']:
            audio_queue = audio_queue[:options['limit']]
        print("=> Queued:", len(audio_queue))

        if len(audio_queue):
            for song in audio_queue:
                print("\nNext up:", song.title, song.spotify_id)
                fetch_wrapper(song)

    def prune(self):
        _, bucket = utils.get_cloud_storage()
        blobs = bucket.list_blobs(prefix='data/audio/')
        all = False
        for blob in blobs:
            if not Song.objects.filter(audio_file=blob.name).exists():
                y = input(f'Delete: {blob.name}? (Y/n/all) ') if not all else 'Y'
                if y == 'all':
                    all = True
                if y == 'Y' or all:
                    blob.delete()


def fetch_wrapper(song):
    try:
        return utils.fetch_audio(song)
    except Exception as err:
        print("Error fetching audio", err, song.pk, song.title)
        return None

