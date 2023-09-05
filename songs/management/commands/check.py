import argparse
from django.core.management.base import BaseCommand
from django.db.models import Q
from songs.models import Song
from songisms import utils


class Command(BaseCommand):
    help = 'Check stuff'

    def add_arguments(self, parser):
        parser.add_argument('--audio', '-a', action=argparse.BooleanOptionalAction)
        parser.add_argument('--stems', '-s', action=argparse.BooleanOptionalAction)
        parser.add_argument('--new', '-n', action=argparse.BooleanOptionalAction)
        parser.add_argument('--lyrics', '-l', action=argparse.BooleanOptionalAction)
        parser.add_argument('--writers', '-w', action=argparse.BooleanOptionalAction)
        parser.add_argument('--rhymes', '-r', action=argparse.BooleanOptionalAction)
        parser.add_argument('--rhymes-ok', '-R', action=argparse.BooleanOptionalAction)
        parser.add_argument('--style', '-S', action=argparse.BooleanOptionalAction)
        parser.add_argument('--metadata', '-m', action=argparse.BooleanOptionalAction)
        parser.add_argument('--transcripts', '-t', action=argparse.BooleanOptionalAction)
        parser.add_argument('--duplicates', '-d', action=argparse.BooleanOptionalAction)

    def handle(self, *args, **options):
        songs = Song.objects.all()
        check_new, check_audio, check_stems, check_lyrics, check_writers, check_rhymes, \
        check_rhymes_ok, check_style, check_metadata, check_transcript, \
        check_duplicates = \
            [options[k] for k in ('new', 'audio', 'stems', 'lyrics', 'writers', \
                                  'rhymes', 'rhymes_ok', 'style', 'metadata',
                                  'transcripts', 'duplicates')]

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
            if check_rhymes_ok:
                self._check_rhymes_ok(song)
            if check_lyrics or check_style:
                if not song.lyrics.strip():
                    print("[NO LYRICS]", song.spotify_id, song.title)
                if check_style:
                    if "\r" in song.lyrics:
                        print("[STYLE: CRs]", song.spotify_id, song.title)
                    if "\n\n" not in song.lyrics:
                        print("[STYLE: NO SECTIONS?]", song.spotify_id, song.title)
                    if "..." in song.lyrics:
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
            if check_duplicates:
                dups = Song.objects.filter(title__istartswith=song.title).filter(
                    Q(artists__in=song.artists.all()) |
                    Q(artists=None) |
                    Q(is_new=True)
                ).exclude(id=song.id)
                if dups.exists():
                    print('\t[DUPLICATES?]', song.spotify_id, song.title)
                    for dup in dups:
                        print('\t\t', dup.spotify_id, dup.title, dup.artists.all())

    def _check_rhymes_ok(self, song):
        from rhymes.nn import predict, load_model
        if not getattr(self, '_model', None):
            self._model, self._scorer = load_model()

        suspect = []
        rhyme_sets = song.rhymes_raw and song.rhymes_raw.split('\n') or []
        for rset in rhyme_sets:
            pairs = utils.get_rhyme_pairs(rset)
            for w1, w2 in pairs:
                score = predict(w1, w2, model=self._model, scorer=self._scorer)
                pred = score >= 0.5
                if not pred:
                    suspect.append((w1, w2, score))

        if len(suspect):
            print("\n[SUSPECT RHYMES]", song.spotify_id, song.title)
            for w1, w2, score in suspect:
                print("\t*", w1, '/', w2, f"[{score:.2f}]",
                      f"({utils.get_ipa_text(w1)} / {utils.get_ipa_text(w2)})")
