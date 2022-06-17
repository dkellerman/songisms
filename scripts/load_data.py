#!/usr/bin/env python ./manage.py script

import datetime
from django.utils import timezone
from django.db import transaction
from api.models import Song
from api.cloud_utils import get_firestore

db = get_firestore()
songs_coll = db.collection('songs')
md_coll = db.collection('metadata')


def main():
    with transaction.atomic():
        for doc in songs_coll.stream():
            id = doc.id
            doc = doc.to_dict()

            mddoc = db.document(f'metadata/{id}').get()
            if mddoc.exists:
                mddoc = mddoc.to_dict()
            else:
                mddoc = None

            spotify_id = doc.get('spotifyId')
            audio_file_path = Song(spotify_id=spotify_id).audio_file_path
            audio_file_exists = Song(spotify_id=spotify_id).audio_file_exists()

            created = doc.get('created')
            if created:
                created = timezone.make_aware(datetime.datetime.fromtimestamp(int(created) / 1e3))
            else:
                created = timezone.now()

            updated = doc.get('updated')
            if updated:
                updated = timezone.make_aware(datetime.datetime.fromtimestamp(int(updated) / 1e3))
            else:
                updated = timezone.now()

            rhymes_raw = '\n'.join(doc.get('rhymes') or [])

            song, created = Song.objects.update_or_create(
                spotify_id=spotify_id,
                defaults=dict(
                    title=doc.get('title'),
                    lyrics=doc.get('content'),
                    lyrics_raw=doc.get('rawContent'),
                    lyrics_ipa=doc.get('ipaContent'),
                    rhymes_raw=rhymes_raw,
                    jaxsta_id=doc.get('jaxstaId'),
                    youtube_id=doc.get('youtubeId'),
                    audio_file=audio_file_path if audio_file_exists else None,
                    metadata=mddoc,
                    created=created,
                    updated=updated,
                )
            )

            print(id, doc.get('title'),
                '[audio]' if audio_file_exists else '',
                '[created]' if created else '[updated]')

            # update m2m fields
            song.set_artists(doc.get('artist'))
            song.set_writers(doc.get('writers'))
            song.set_song_tags(doc.get('tags'))
            song.set_snippet_tags(doc.get('snippetTags'))


if __name__ == '__main__':
    main()
