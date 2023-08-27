'''Utilities for various APIs and cloud services'''

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

google_key = json.loads(base64.b64decode(os.environ['SISM_GOOGLE_CREDENTIALS']))
storage_credentials = service_account.Credentials.from_service_account_info(google_key)
sclient = SClient(credentials=storage_credentials)
bucket = sclient.bucket(settings.GS_BUCKET_NAME)


def get_cloud_storage():
    return sclient, bucket


def get_storage_blob(fname):
    return bucket.blob(fname)


def fetch_audio(song, convert=False):
    '''Fetch audio for a song from youtube. Convert to mp3 if convert=True, requires ffmpeg'''
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
    tmpfile_mp3 = None

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
            ffmpeg.input(tmpfile).output(
                tmpfile_mp3, ac=1, audio_bitrate='128k').run()
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
        if tmpfile and os.path.exists(tmpfile):
            os.remove(tmpfile)
        if tmpfile_mp3 and os.path.exists(tmpfile_mp3):
            os.remove(tmpfile_mp3)
    except:
        print("problem removing temp files", tmpfile, tmpfile_mp3)

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
    from api.models import Attachment
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
