import sh
import datetime
import argparse
from django.core.management.base import BaseCommand
from songs.models import Song
from songisms import utils


class Command(BaseCommand):
    '''Requires env vars set:
        SPOTIPY_CLIENT_ID
        SPOTIPY_CLIENT_SECRET
        SPOTIPY_REDIRECT_URI (?)

        Usage:
        ./manage.py playlist --sync --pid PLAYLIST_ID

        Writes DB & playlist backups to data/backup/ directory
    '''

    help = 'Sync master playlist and songs DB'
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_arguments(self, parser):
        parser.add_argument('--sync', '-s', action=argparse.BooleanOptionalAction)
        parser.add_argument('--restore', '-r', type=str, default=None)
        parser.add_argument('--pid', '-p', type=str, default="35EttZV5qLKtQZgxxOmSGN")

    def handle(self, *args, **options):
        self.playlist_id = options['pid']
        if options['sync']:
            self.backup()
            self.sync_playlist()
        elif options['restore']:
            self.restore_playlist(options['restore'])

    def sync_playlist(self):
        existing_playlist_ids = utils.get_playlist_track_ids(self.playlist_id)
        print("Existing tracks on playlist:", len(existing_playlist_ids))
        with open(f'./data/backup/playlist__{self.ts}.txt', 'w') as f:
            f.write('\n'.join(existing_playlist_ids))

        songs_not_on_playlist = []
        songs_not_in_db = []

        for song in Song.objects.exclude(spotify_id=None):
            if song.spotify_id not in existing_playlist_ids:
                songs_not_on_playlist.append(song.spotify_id)
        for id in existing_playlist_ids:
            if not Song.objects.filter(spotify_id=id).exists():
                songs_not_in_db.append(id)

        print("Potential new tracks for playlist:", len(songs_not_on_playlist))
        add_to_playlist = self.confirm_tracks("Add to playlist", songs_not_on_playlist)

        if len(add_to_playlist) > 0:
            utils.add_songs_to_playlist(self.playlist_id, [t[0] for t in add_to_playlist])
            print(len(add_to_playlist), "songs added to playlist")

        print("Potential new tracks for DB:", len(songs_not_in_db))
        add_to_db = self.confirm_tracks("Add to DB", songs_not_in_db)
        rm_from_playlist = set(songs_not_in_db) - set([ t[0] for t in add_to_db])

        for id, track in add_to_db:
            print("Creating song:", track['name'], "id:", id)
            song = Song.objects.create(spotify_id=id, title=track['name'], is_new=True)
            song.set_artists([ a['name'] for a in track['artists'] ])
            song.save()

        if len(rm_from_playlist) > 0:
            resp = input(f"Remove {len(rm_from_playlist)} song(s) from playlist? (Y/n): ")
            if resp == 'Y':
                utils.remove_songs_from_playlist(self.playlist_id, list(rm_from_playlist))

    def restore_playlist(self, filename):
        ids = open(filename, 'r').read().split('\n')
        existing_playlist_ids = utils.get_playlist_track_ids(self.playlist_id)
        print("Existing tracks on playlist:", len(existing_playlist_ids))
        to_add = list(set(ids) - set(existing_playlist_ids))
        print("New tracks for playlist:", len(to_add))
        resp = input("Ok? (Y/n): ")
        if resp == 'Y':
            utils.add_songs_to_playlist(self.playlist_id, to_add)

    def backup(self):
        sh.mkdir('-p', './data/backup')
        sh.pg_dump('-a', '-d', 'songisms2', '-f', f'./data/backup/songisms__{self.ts}.sql')

    def confirm_tracks(self, action, track_ids):
        all = False
        confirmed = []
        for id in track_ids:
            track = utils.get_track(id)
            artists = ','.join([artist['name'] for artist in track['artists']])
            if all:
                answer = 'Y'
            else:
                answer = input(f"{action}: {track['name']} by {artists} id: {id}? (Y/n/all/end): ")
                if answer.lower() == 'end':
                    return confirmed
                if answer.lower() == 'all':
                    all = True
                    answer = 'Y'
            if answer == 'Y':
                confirmed.append((id, track))
        return confirmed
