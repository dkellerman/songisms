import argparse
import itertools
import gc
from itertools import product
from tqdm import tqdm
from django.core.management.base import BaseCommand
from django.db import transaction
from nltk import FreqDist
from rhymes.models import NGram, Rhyme, SongNGram, Cache, Vote
from songisms import utils


class Command(BaseCommand):
    help = 'Process ngrams and rhymes'

    def add_arguments(self, parser):
        parser.add_argument('--rescore', '-s', action=argparse.BooleanOptionalAction)
        parser.add_argument('--dry-run', '-D', action=argparse.BooleanOptionalAction)
        parser.add_argument('--batch-size', '-b', type=int, default=1000)


    def handle(self, *args, **options):
        batch_size, dry_run, rescore = [options[k] for k in ('batch_size', 'dry_run', 'rescore',)]

        try:
            from songs.models import Song  # songs app needs to be in INSTALLED_APPS
        except:
            print('songs app not installed')
            return

        ngrams = dict()
        rhymes = dict()
        song_ngrams = dict()
        vector_cache, _ = Cache.objects.get_or_create(key='ngram_vector')

        # prep rhyme votes reference
        votes = Vote.objects.filter(label__in=['good', 'bad'])
        positive = dict()
        negative = dict()
        for vote in tqdm(votes, desc='creating votes index'):
            ukey = '_'.join(sorted([vote.anchor, vote.alt1]))
            if vote.label == 'good':
                positive[ukey] = positive.get(ukey, 0) + 1
            else:
                negative[ukey] = negative.get(ukey, 0) + 1

        # loop through all songs with lyrics
        songs = Song.objects.filter(is_new=False).exclude(lyrics=None)

        for song in tqdm(songs, desc='lyric ngrams'):
            # lyric ngrams
            texts = utils.get_lyric_ngrams(song.lyrics, range(5))
            for text, n in texts:
                ngrams[text] = ngrams.get(text, None) or dict(text=text, n=n)
                song_ngram = song_ngrams.get((song.spotify_id, text), None)
                # update song_ngrams and count
                if song_ngram:
                    song_ngram['count'] += 1
                else:
                    song_ngrams[(song.spotify_id, text)] = dict(ngram=ngrams[text],
                                                                song_uid=song.spotify_id, count=1)

            # update song_count for unique ngrams
            for text, _ in list(set(texts)):
                ngrams[text]['song_count'] = ngrams[text].get('song_count', 0) + 1

            # song rhymes
            if song.rhymes_raw:
                rhyme_pairs = utils.get_rhyme_pairs(song.rhymes_raw)
                for from_text, to_text in rhyme_pairs:
                    ukey = '_'.join(sorted([from_text, to_text]))
                    if ukey in negative:
                        continue
                    ngrams[from_text] = ngrams.get(from_text) or dict(text=from_text, n=len(from_text.split()))
                    ngrams[to_text] = ngrams.get(to_text) or dict(text=to_text, n=len(to_text.split()))
                    rhymes[(from_text, to_text, song.spotify_id)] = \
                        dict(from_ngram=ngrams[from_text], to_ngram=ngrams[to_text],
                             song_uid=song.spotify_id, level=1, ukey=ukey)

        # add some rhymes data from muse for common words
        single_words = [n for n in ngrams.values() if n['n'] == 1]
        muse_rhymes = set()

        for from_ngram in tqdm(single_words, desc='datamuse rhymes'):
            vals = utils.get_datamuse_rhymes(from_ngram['text'], cache_only=False)
            for val in vals:
                to_text = val['word']
                from_text = from_ngram['text']
                if to_text in utils.data.common_words and from_text in utils.data.common_words:
                    ngrams[to_text] = ngrams.get(to_text) or dict(text=to_text, n=len(to_text.split()))
                    rkey = (from_text, to_text, None)
                    if rkey not in rhymes:
                        ukey = '_'.join(sorted([from_text, to_text]))
                        rhymes[rkey] = dict(from_ngram=ngrams[from_text], to_ngram=ngrams[to_text],
                                            song_uid=None, level=1, ukey=ukey)
                        muse_rhymes.add(to_text)

        # create rhymes lookup index for later
        rhymes_index = dict()
        for r in tqdm(rhymes.values(), desc='indexing rhymes'):
            rhymes_index[r['from_ngram']['text']] = list(set(
                [ r2['to_ngram']['text'] for r2 in rhymes.values()
                  if r2['from_ngram']['text'] == r['from_ngram']['text'] ]
            ))

        # imply some multi-word rhymes
        multirhyme_candidates = [
            n['text'] for n in tqdm(ngrams.values(), desc='prepping multi rhymes')
                if (n.get('song_count', 0) > 2)
                and (n['n'] in (2, 3,))
                and (utils.get_mscore(n['text']) > 3)
                and (not is_repeated(n['text']))]

        for ngram in tqdm(multirhyme_candidates, desc='making multi rhymes'):
            grams = ngram.split()
            lists = []
            for gram in grams:
                to_rhymes = rhymes_index.get(gram, [])
                lists.append(to_rhymes)
            combos = [p for p in product(*lists)]
            for combo in combos:
                val = ' '.join(combo)
                entry = ngrams.get(val)
                if (entry
                    and (entry.get('song_count', 0) > 2)
                    and (utils.get_mscore(val) > 3)
                    and (not is_repeated(val))
                ):
                    rkey = (ngram, val, None)
                    if rkey not in rhymes:
                        ukey = '_'.join(sorted([ngram, val]))
                        rhymes[rkey] = dict(from_ngram=ngrams[ngram], to_ngram=ngrams[val],
                                            song_uid=None, level=3, ukey=ukey)

        # make level 2 rhymes (aka rhymes of rhymes)
        level1_rhymes = list(rhymes.values())
        for l1 in tqdm(level1_rhymes, desc='level 2 rhymes'):
            for l2 in [r for r in level1_rhymes if r['from_ngram']['text'] == l1['to_ngram']['text']]:
                rkey = (l1['from_ngram']['text'], l2['to_ngram']['text'], None)
                if rkey not in rhymes:
                    ukey = '_'.join(sorted([l1['from_ngram']['text'], l2['to_ngram']['text']]))
                    rhymes[rkey] = dict(from_ngram=l1['from_ngram'], to_ngram=l2['to_ngram'],
                                        song_uid=None, level=2, ukey=ukey)

        # get feature vectors
        for ngram in tqdm(ngrams.values(), desc='ngram vectors'):
            ngram['phones'] = vector_cache.get(ngram['text'], utils.get_vowel_vector) or None

        # get meaning scores
        for ngram in tqdm(ngrams.values(), desc='ngram mscores'):
            ngram['mscore'] = utils.get_mscore(ngram['text'])

        # get IPA
        ipa_cache, _ = Cache.objects.get_or_create(key='ngram_ipa')
        for ngram in tqdm(ngrams.values(), desc='ngram ipa'):
            ngram['ipa'] = ipa_cache.get(ngram['text'], utils.get_ipa_text)

        # get stresses vector
        stresses_cache, _ = Cache.objects.get_or_create(key='ngram_stresses')
        for ngram in tqdm(ngrams.values(), desc='ngram stresses'):
            ngram['stresses'] = stresses_cache.get(ngram['text'], utils.get_stresses_vector)

        # create an index counts per ngram length for use in the next step
        n_counts = [0 for _ in range(15)]
        for sn in tqdm(song_ngrams.values(), desc='index ngram counts'):
            n = sn['ngram']['n']
            ct = sn['count']
            n_counts[n - 1] = (n_counts[n - 1] or 0) + ct
            ngram['count'] = (sn['ngram'].get('count') or 0) + ct
            ngrams[sn['ngram']['text']]['count'] = ngram['count']
        single_word_ct = n_counts[0]
        song_ct = songs.count()
        # frequency of ngrams appearing in titles
        title_ngrams = FreqDist(itertools.chain(*[n[0] for n in [utils.get_lyric_ngrams(s.title) for s in songs]]))

        # now calculate various ngram percentages
        # * pct = ngram count as percentage of all ngram occurrences (with same n)
        # * adj_pct = percentage of a multiword phrase appearing above chance
        # * song_pct = percentage of songs with this ngram
        # * title_pct = percentage of song titles this ngram appears in
        for ngram in tqdm(ngrams.values(), desc='ngrams pct'):
            if ngram['n'] > 1:
                subgrams = [ngrams[gram] for gram in ngram['text'].split() if gram and (gram in ngrams)]
                total_with_same_n = n_counts[ngram['n'] - 1]
                ngram_pct = float((ngram.get('count') or 0.0) / total_with_same_n)
                chance_pct = 1.0
                for gram in subgrams:
                    gram_pct = float((gram.get('count') or 0.0) / single_word_ct)
                    chance_pct *= gram_pct
                ngram['pct'] = ngram_pct
                ngram['adj_pct'] = ngram_pct - chance_pct
            else:
                ngram['pct'] = float((ngram.get('count') or 0.0) / single_word_ct)
                ngram['adj_pct'] = ngram['pct']
            ngram['title_pct'] = title_ngrams.freq(ngram['text'])
            ngram['song_pct'] = ngram.get('song_count', 0) / song_ct


        # scores from nn model
        scores_cache, _ = Cache.objects.get_or_create(key='rhyme_scores')
        scores = []
        if rescore:
            scores_cache.clear()
        to_del = set()
        for rhyme in tqdm(rhymes.values(),
                          f"{'generating' if rescore else 'using cached'} "
                          "rhyme scores"):
            rfrom = rhyme['from_ngram']['text']
            rto = rhyme['to_ngram']['text']
            ukey = '_'.join(sorted([rfrom, rto]))
            if ukey in negative:
                to_del.add(ukey)
                continue
            rhyme['score'] = scores_cache.get(ukey, get_score if rescore else None)
            if rhyme['score']:
                scores.append(rhyme['score'])

        # remove flagged rhymes
        rhymes = {key: obj for key, obj in rhymes.items() if obj['ukey'] not in to_del}

        # print some counts
        print('ngrams', len(ngrams.values()))
        print('rhymes', len(rhymes.values()))
        print('song_ngrams', len(song_ngrams.values()))
        print('avg rhyme score', sum(scores) / len(scores))

        if dry_run:
            return

        # save the db caches
        if rescore:
            scores_cache.save()
        utils.get_datamuse_cache().save()
        for c in tqdm([vector_cache, ipa_cache, stresses_cache], desc='saving db caches'):
            c.save()
            del c
        del title_ngrams
        gc.collect()

        # begin the DB writes
        with transaction.atomic():
            print('deleting...')
            Rhyme.objects.all().delete()
            NGram.objects.all().delete()
            SongNGram.objects.all().delete()

            # NGram (rhymes_ngram)
            ngrams = [NGram(**{k: v for k, v in n.items()})
                      for n in tqdm(ngrams.values(), desc='prepping ngrams')]
            print('writing ngrams', len(ngrams))
            ngrams = NGram.objects.bulk_create(ngrams, batch_size=batch_size)
            ngrams = dict([(n.text, n) for n in ngrams])

            # Rhyme (rhymes_rhyme)
            rhyme_objs = []
            for rhyme in tqdm(rhymes.values(), desc='prepping rhymes'):
                nfrom = ngrams[rhyme['from_ngram']['text']]
                nto = ngrams[rhyme['to_ngram']['text']]
                song_uid = rhyme['song_uid']
                rhyme_objs.append(Rhyme(from_ngram=nfrom, to_ngram=nto, song_uid=song_uid,
                                        level=rhyme['level'], score=rhyme['score']))
                # write the reverse rhyme as well
                revkey = (nto.text, nfrom.text, song_uid if song_uid else None)
                if revkey not in rhymes:
                    rhyme_objs.append(Rhyme(from_ngram=nto, to_ngram=nfrom, song_uid=song_uid,
                                            level=rhyme['level'], score=rhyme['score']))

            print('writing rhymes', len(rhyme_objs))
            Rhyme.objects.bulk_create(rhyme_objs, batch_size=batch_size)
            del rhyme_objs
            gc.collect()

            # SongNGram (rhymes_songngram)
            sn_objs = []
            for sn in tqdm(song_ngrams.values(), desc='prepping song_ngrams'):
                n = ngrams[sn['ngram']['text']]
                sn_objs.append(SongNGram(song_uid=sn['song_uid'], ngram=n, count=sn['count']))
            print('creating song_ngrams', len(sn_objs))
            SongNGram.objects.bulk_create(sn_objs, batch_size=batch_size)

            print('finishing transaction')
        print('done')


def is_repeated(w):
    return len(set(w.split())) < len(w.split())


_score_model, _scorer = None, None

def get_score(key):
    from rhymes.nn import predict, load_model
    global _score_model, _scorer

    if not _score_model:
        _score_model, _scorer = load_model()

    w1, w2 = key.split('_')
    return predict(w1, w2, model=_score_model, scorer=_scorer)
