'''Utilities for various APIs and cloud services'''

import os
import time
import json
import base64
import requests
import shutil
import requests
import math
from functools import lru_cache
from urllib.parse import urlencode
from django.core.files import File
from django.conf import settings
from google.oauth2 import service_account

google_key = json.loads(base64.b64decode(os.environ['SISM_GOOGLE_CREDENTIALS']))
storage_credentials = service_account.Credentials.from_service_account_info(google_key)
storage_client = None
bucket = None


def get_cloud_storage():
    global storage_client, bucket
    from google.cloud.storage import Client as SClient
    if storage_client is None:
        storage_client = SClient(credentials=storage_credentials)
        bucket = storage_client.bucket(settings.GS_BUCKET_NAME)
    return storage_client, bucket


def get_storage_blob(fname):
    if bucket is None:
        return None
    return bucket.blob(fname)


def fetch_audio(song):
    '''Fetch audio for a song from youtube
    '''
    from pytube import YouTube

    yt_id = song.youtube_id
    print("***** fetching audio", song.id, song.title, yt_id)

    yt = YouTube(f'https://www.youtube.com/watch?v={yt_id}')
    audio = yt.streams.filter(only_audio=True).first()

    yt_meta = dict(
        id=yt_id,
        song_id=song.spotify_id,
        created=time.time(),
        author=yt.author,
        duration=yt.length,
        thumb=yt.thumbnail_url,
        title=yt.title,
        viewcount=yt.views,
        watchv_url=yt.watch_url,
        description=yt.description,
        keywords=yt.keywords,
    )

    md = song.metadata or dict()
    md['youtube'] = yt_meta
    song.metadata = md

    ext = audio.get_file_path().split('.')[-1]
    fname = f'{song.spotify_id}.{ext}'
    tmpfile = f'/tmp/{fname}'

    if not os.path.exists(tmpfile):
        print('download', tmpfile)
        audio.download(filename=fname, output_path='/tmp/')
    else:
        print('audio file exists', tmpfile)

    fname_upload = fname
    tmpfile_upload = tmpfile

    if os.path.exists(tmpfile_upload):
        print('uploading audio', tmpfile_upload)
        with open(tmpfile_upload, 'rb') as f:
            song.audio_file.save(fname_upload, File(f))
            song.save()
    else:
        print('missing upload file', tmpfile_upload)

    try:
        if tmpfile and os.path.exists(tmpfile):
            os.remove(tmpfile)
    except:
        print("problem removing temp files", tmpfile)

    print("done")


MOISES_WORKFLOWS = {
    'stems': 'moises/stems-vocals-accompaniment',
    'transcript': 'sism2',
}


def queue_workflow(workflow_id, song):
    '''Start moises.ai workflow
    '''
    resp = requests.post('https://developer-api.moises.ai/api/job', json={
        'name': f'Workflow {workflow_id} for {song.spotify_id}',
        'workflow': MOISES_WORKFLOWS[workflow_id],
        'params': {
            'inputUrl': song.audio_file_url,
        }
    }, headers={
        'Authorization': f'{settings.MOISES_API_KEY}',
        'Content-Type': 'application/json; charset=utf-8'
    })
    if resp.ok:
        data = resp.json()
        id = data['id']
        print(song.spotify_id, song.title, id)
    else:
        print('error', resp.text)
        return None
    return id


def fetch_workflow_by_id(song, id):
    from songs.models import Attachment
    resp = requests.get(f'https://developer-api.moises.ai/api/job/{id}', headers={
        'Authorization': f'{settings.MOISES_API_KEY}'
    })
    if resp.ok:
        data = resp.json()
        if data['status'] == 'SUCCEEDED':
            attachments = []
            for key, url in data['result'].items():
                fname = f'{song.spotify_id}__{key}'
                fpath = f'/tmp/{fname}'
                if os.path.exists(fpath):
                    os.remove(fpath)

                print('\t=> fetching', key, url, '=>', fpath)
                fresp = requests.get(url, stream=True)
                with open(fpath, 'wb') as tmpfile:
                    shutil.copyfileobj(fresp.raw, tmpfile)

                a = song.get_attachment(key)
                if a:
                    a.delete()
                a = Attachment(content_object=song, attachment_type=key)
                with open(fpath, 'rb') as f:
                    a.file.save(fname, File(f))
                os.remove(fpath)
                attachments.append(a)

            return attachments

        elif data['status'] == 'FAILED':
            print('[FAILED]', id, song.title)
            return True

        else:
            print('[PENDING]', id, song.title)
            return None
    else:
        raise Exception(f'fetch workflow failed {resp.text}')


def fetch_workflow(workflow_id, song):
    id = queue_workflow(workflow_id, song)
    attachments = None

    while True:
        attachments = fetch_workflow_by_id(song, id)
        if not attachments:
            time.sleep(20)
        else:
            break

    if attachments is True:
        print("[FAILED]", song.id, song.spotify_id, song.title)
    else:
        for a in attachments or []:
            print("[FINISHED]", a.attachment_type, a.file.url)

    return attachments


@lru_cache(maxsize=None)
def get_datamuse_cache():
    from rhymes.models import Cache
    cache, _ = Cache.objects.get_or_create(key='datamuse')
    return cache


def get_datamuse_rhymes(key, cache_only=False):
    return get_datamuse_cache().get(key, fetch_datamuse_rhymes if not cache_only else None)


def fetch_datamuse_rhymes(key):
    query = urlencode(dict(rel_rhy=key, max=50))
    query2 = urlencode(dict(rel_nry=key, max=50))
    vals = []

    print("Fetch datamuse rhymes", query, "AND", query2)
    try:
        vals += requests.get(f'https://api.datamuse.com/words?{query}').json()
    except:
        print('error retrieving datamuse RHY for', key)
    try:
        vals += requests.get(f'https://api.datamuse.com/words?{query2}').json()
    except:
        print('error retrieving datamuse NRY for', key)

    return vals


@lru_cache(maxsize=None)
def get_spotify_client():
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    spot = spotipy.Spotify(auth_manager=SpotifyOAuth(scope='playlist-modify-private'))
    return spot


def get_playlist_track_ids(playlist_id, limit=1000):
    spot = get_spotify_client()
    existing_playlist_ids = []
    for i in range(0, math.ceil(limit / 100)):
        tracks = spot.playlist_tracks(playlist_id, offset=i * 100, limit=100)
        existing_playlist_ids += [track['track']['id'] for track in tracks['items']]
        if len(tracks['items']) < 100:
            break

    return existing_playlist_ids


def add_songs_to_playlist(playlist_id, song_ids):
    spot = get_spotify_client()
    spot.playlist_add_items(playlist_id, song_ids)


def remove_songs_from_playlist(playlist_id, song_ids):
    spot = get_spotify_client()
    spot.playlist_remove_all_occurrences_of_items(playlist_id, song_ids)


def get_track(id):
    spot = get_spotify_client()
    track = spot.track(id)
    return track
