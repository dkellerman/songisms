import argparse
import itertools
from tqdm import tqdm
from django.core.management.base import BaseCommand
from django.db import transaction
from nltk import FreqDist
from rhymes.models import NGram, Rhyme, Vote
from songisms import utils


class Command(BaseCommand):
    help = 'Process ngrams and rhymes'

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', '-D', action=argparse.BooleanOptionalAction)


    def handle(self, *args, **options):
        dry_run = [options[k] for k in ('dry_run',)]

        try:
            from songs.models import Song  # songs app needs to be in INSTALLED_APPS
        except:
            print('songs app not installed')
            return

        ngrams = dict()
        rhymes = dict()
        up_votes = dict()
        down_votes = dict()

        # gather votes
        for vote in Vote.objects.all():
            uid1 = '_'.join(sorted([vote.anchor, vote.alt1]))
            uid2 = '_'.join(sorted([vote.anchor, vote.alt2])) if vote.alt2 else None
            if vote.label == 'good':
                up_votes[uid1] = up_votes.get(uid1, 0) + 1
            elif vote.label == 'bad':
                down_votes[uid1] = down_votes.get(uid1, 0) + 1
            elif vote.label == 'both':
                up_votes[uid1] = up_votes.get(uid1, 0) + 1
                up_votes[uid2] = up_votes.get(uid2, 0) + 1
            elif vote.label == 'neither':
                down_votes[uid1] = down_votes.get(uid1, 0) + 1
                down_votes[uid2] = down_votes.get(uid2, 0) + 1

        # loop through all songs
        songs = Song.objects.filter(is_new=False).exclude(rhymes_raw=None)

        song_ngrams = set()
        for song in tqdm(songs, desc='parse ngrams'):
            # lyric ngrams
            texts = utils.get_lyric_ngrams(song.lyrics, range(5))
            for text, n in texts:
                ngrams[text] = ngrams.get(text, None) or make_ngram(text)
                ngrams[text].frequency += 1
                song_ngrams.add((text, song.pk))

        song_rhymes = set()
        for song in tqdm(songs.exclude(rhymes_raw=None), desc='parse rhymes'):
            # song rhymes
            rhyme_pairs = utils.get_rhyme_pairs(song.rhymes_raw)
            for from_text, to_text in rhyme_pairs:
                from_text = utils.normalize_lyric(from_text)
                to_text = utils.normalize_lyric(to_text)
                uid = '_'.join(sorted([from_text, to_text]))
                ngrams[from_text] = ngrams.get(from_text) or make_ngram(from_text)
                ngrams[to_text] = ngrams.get(to_text) or make_ngram(to_text)
                rhymes[uid] = rhymes.get(uid) or Rhyme(
                    uid=uid,
                    from_ngram=ngrams[from_text],
                    to_ngram=ngrams[to_text],
                    source='song',
                    score=get_score(uid),
                    uscore=up_votes.get(uid, 0) - down_votes.get(uid, 0),
                    frequency=0,
                    song_ct=0)
                rhymes[uid].frequency += 1
                song_rhymes.add((uid, song.pk))

        for uid, _ in list(song_rhymes):
            rhymes[uid].song_ct += 1

        add_ngram_stats(ngrams, songs, song_ngrams)

        print('rhymes', len(rhymes))
        print('ngrams', len(ngrams))
        print('up votes', len(up_votes))
        print('down votes', len(down_votes))

        if dry_run:
            return

        # begin the DB writes
        with transaction.atomic():
            print('writing ngrams...')
            NGram.objects.all().delete()
            ngrams = NGram.objects.bulk_create(ngrams.values())

            print('writing rhymes...')
            Rhyme.objects.all().delete()
            Rhyme.objects.bulk_create(rhymes.values())

            print('finishing transaction...')
        print('done')


def add_ngram_stats(ngrams, songs, song_ngrams):
    # create an index with counts per ngram length
    n_counts = [0 for _ in range(15)]
    for ngram in ngrams.values():
        n_counts[ngram.n - 1] = n_counts[ngram.n] + ngram.frequency
    single_word_ct = n_counts[0]

    # ngram song counts
    song_cts = dict()
    for text, _ in song_ngrams:
        song_cts[text] = song_cts.get(text, 0) + 1

    # frequency of ngrams appearing in titles
    title_ngrams = FreqDist(itertools.chain(*[n[0] for n in [utils.get_lyric_ngrams(s.title)
                                                             for s in songs]]))
    total_song_ct = len(songs)

    # now calculate various ngram percentages
    # * pct = ngram count as percentage of all ngram occurrences (with same n)
    # * adj_pct = percentage of a multiword phrase appearing above chance
    # * song_pct = percentage of songs with this ngram
    # * title_pct = percentage of song titles this ngram appears in
    for ngram in tqdm(ngrams.values(), desc='ngram stats'):
        if ngram.n > 1:
            subgrams = [ngrams[gram] for gram in ngram.text.split() if gram and (gram in ngrams)]
            total_with_same_n = n_counts[ngram.n - 1]
            ngram.pct = float(ngram.frequency / total_with_same_n)
            chance_pct = 1.0
            for subgram in subgrams:
                gram_pct = float(subgram.frequency / single_word_ct)
                chance_pct *= gram_pct
            ngram.adj_pct = ngram.pct - chance_pct
        else:
            ngram.pct = float(ngram.frequency / single_word_ct)
            ngram.adj_pct = ngram.pct
        ngram.title_pct = title_ngrams.freq(ngram.text)
        ngram.song_pct = float(song_cts.get(ngram.text, 0) / total_song_ct)


_score_model, _scorer = None, None

def get_score(key):
    from rhymes.nn import predict, load_script_model
    global _score_model, _scorer

    if not _score_model:
        _score_model, _scorer = load_script_model()

    w1, w2 = key.split('_')
    return predict(w1, w2, model=_score_model, scorer=_scorer)


def make_ngram(text):
    return NGram(
        text=text,
        n=len(text.split()),
        mscore=utils.get_mscore(text),
        frequency=0,
    )
