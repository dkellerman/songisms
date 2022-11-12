import os
import time
import pafy
import json
import base64
import requests
import shutil
from django.core.files import File
from django.conf import settings
from google.cloud.storage import Client as SClient
from google.oauth2 import service_account

key = json.loads(base64.b64decode(os.environ['SISM_GOOGLE_CREDENTIALS']))
storage_credentials = service_account.Credentials.from_service_account_info(key)
sclient = SClient(credentials=storage_credentials)
bucket = sclient.bucket(settings.GS_BUCKET_NAME)


def get_cloud_storage():
    return sclient, bucket


def get_storage_blob(fname):
    return bucket.blob(fname)


def fetch_audio(song, convert=False):
    yt_id = song.youtube_id
    print("***** fetching video", song.id, song.title, yt_id)

    video = pafy.new(f'https://www.youtube.com/watch?v={yt_id}')
    yt_meta = dict(
        id=yt_id,
        song_id=song.id,
        created=time.time(),
        author=video.author,
        bigthumb=video.bigthumb,
        bigthumbhd=video.bigthumbhd,
        category=video.category,
        dislikes=video.dislikes,
        duration=video.duration,
        expiry=video.expiry,
        likes=video.likes,
        thumb=video.thumb,
        title=video.title,
        viewcount=video.viewcount,
        watchv_url=video.watchv_url,
        # description=video.description,
        # keywords
    )

    md = song.metadata or dict()
    md['youtube'] = yt_meta
    song.metadata = md

    audio = video.getbestaudio()
    fname = f'{song.spotify_id}.{audio.extension}'
    tmpfile = f'/tmp/{fname}'

    if not os.path.exists(tmpfile):
        print('download', tmpfile)
        audio.download(filepath=tmpfile, quiet=False)
    else:
        print('webm file exists')

    if convert:
        import ffmpeg
        fname_mp3 = f'{yt_id}.mp3'
        tmpfile_mp3 = f'/tmp/{fname_mp3}'
        if not os.path.exists(tmpfile_mp3):
            print('convert to mp3...')
            ffmpeg.input(tmpfile).output(tmpfile_mp3, ac=1, audio_bitrate='128k').run()
        else:
            print('mp3 exists')
        fname_upload = fname_mp3
        tmpfile_upload = tmpfile_mp3
    else:
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
        if os.path.exists(tmpfile):
            os.remove(tmpfile)
        if os.path.exists(tmpfile_mp3):
            os.remove(tmpfile_mp3)
    except:
        print("problem removing temp files", tmpfile, tmpfile_mp3)

    print("done")


def queue_stems(song):
    resp = requests.post('https://developer.moises.ai/api/media', json={
        'inputUrl': song.audio_file_url,
        'operations': [
            dict(type='STEMS', mode='vocals-drums-bass-background_vocals-other')
        ]
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


def fetch_stems_by_id(song, id):
    from api.models import Attachment
    resp2 = requests.get(f'https://developer.moises.ai/api/media/{id}', headers={
        'Authorization': f'{settings.MOISES_API_KEY}'
    })
    if resp2.ok:
        data2 = resp2.json()
        op = data2['operations'][0]
        if op['status'] == 'COMPLETED':
            attachments = []
            for key, url in op['result']['files'].items():
                if key.endswith('HighRes') or Attachment.objects.filter(object_id=song.pk, attachment_type=key).exists():
                    continue

                ext = url.split('.')[-1]
                fname = f'{song.spotify_id}.{key}.{ext}'
                fpath = f'/tmp/{fname}'
                if os.path.exists(fpath):
                    os.remove(fpath)

                print('\t=> fetching', key, url, '=>', fpath)
                fresp = requests.get(url, stream=True)
                with open(fpath, 'wb') as tmpfile:
                    shutil.copyfileobj(fresp.raw, tmpfile)

                a = Attachment(content_object=song, attachment_type=key)
                with open(fpath, 'rb') as f:
                    a.file.save(fname, File(f))
                os.remove(fpath)
                attachments.append(a)
        else:
            print('[PENDING]', id, song.title)
            return None
        return attachments
    else:
        raise Exception(f'fetch stems failed {resp2.text}')


def fetch_stems(song):
    id = queue_stems(song)
    attachments = None

    while True:
        attachments = fetch_stems_by_id(song, id)
        if not attachments:
            time.sleep(30)
        else:
            break

    for a in attachments:
        print(a.attachment_type, a.file.url)

    return attachments
