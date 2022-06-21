#!/usr/bin/env python

import argparse
import gc
from urllib.parse import urlencode
from itertools import product
import requests
from django.core.management.base import BaseCommand
from django.db import transaction
from api.models import *
from api.nlp_utils import get_lyric_ngrams, get_rhyme_pairs, get_common_words, get_mscore, get_ipa
from django.core.cache import cache
from tqdm import tqdm


class Command(BaseCommand):
    help = 'Process text'

    def add_arguments(self, parser):
        parser.add_argument('--caches', '-c', action=argparse.BooleanOptionalAction)
        parser.add_argument('--caches-only', '-C', action=argparse.BooleanOptionalAction)
        parser.add_argument('--dry-run', '-D', action=argparse.BooleanOptionalAction)
        parser.add_argument('--batch-size', '-b', type=int, default=1000)

    def handle(self, *args, **options):
        batch_size, dry_run, do_caches = [options[k] for k in ('batch_size', 'dry_run', 'caches')]
        dry_run = options['dry_run']
        do_caches = options['caches']

        if options['caches_only']:
            reset_caches()
            return

        songs = Song.objects.all()
        ngrams = dict()
        rhymes = dict()
        song_ngrams = dict()

        phones_cache, _ = Cache.objects.get_or_create(key='ngram_phones')
        datamuse_cache, _ = Cache.objects.get_or_create(key='datamuse')
        mscores_cache, _ = Cache.objects.get_or_create(key='ngram_mscores')

        # lyric ngrams
        for idx, song in enumerate(tqdm(songs, desc='lyric ngrams')):
            if song.lyrics:
                texts = get_lyric_ngrams(song.lyrics, range(5))
                for text, n in texts:
                    ngrams[text] = ngrams.get(text, None) or dict(text=text, n=n)
                    ngrams[text]['song_count'] = ngrams[text].get('song_count', 0) + 1
                    song_ngram = song_ngrams.get((song.id, text), None)
                    if song_ngram:
                        song_ngram['count'] += 1
                    else:
                        song_ngrams[(song.id, text)] = dict(ngram=ngrams[text], song=song, count=1)

            if song.rhymes_raw:
                rhyme_pairs = get_rhyme_pairs(song.rhymes_raw)
                for from_text, to_text in rhyme_pairs:
                    ngrams[from_text] = ngrams.get(from_text) or dict(text=from_text, n=len(from_text.split()))
                    ngrams[to_text] = ngrams.get(to_text) or dict(text=to_text, n=len(to_text.split()))
                    rhymes[(from_text, to_text, song.id)] = \
                        dict(from_ngram=ngrams[from_text], to_ngram=ngrams[to_text], song=song, level=1)

        extra = []
        # with open('./data/mine.txt', 'r') as f:
        #     extra += get_lyric_ngrams(f.read(), range(5))
        with open('./data/idioms.txt', 'r') as f:
            extra += get_lyric_ngrams(f.read(), range(5))
        for text, n in tqdm(extra, desc='extra ngrams'):
            ngrams[text] = ngrams.get(text, None) or dict(text=text, n=n)

        n1 = [n for n in ngrams.values() if n['n'] == 1]
        common = get_common_words()
        rmuse = set()

        for from_ngram in tqdm(n1, desc='datamuse rhymes'):
            vals = datamuse_cache.get(from_ngram['text'], fetch_datamuse_rhymes)
            for val in vals:
                to_word = val['word']
                from_word = from_ngram['text']
                if to_word in common and from_word in common:
                    ngrams[to_word] = ngrams.get(to_word) or dict(text=to_word, n=len(to_word.split()))
                    rkey = (from_ngram['text'], to_word, None)
                    if rkey not in rhymes:
                        rhymes[rkey] = dict(from_ngram=from_ngram, to_ngram=ngrams[to_word], song=None, level=1)
                        rmuse.add(to_word)

        ridx = dict()
        for r in tqdm(rhymes.values(), desc='indexing rhymes'):
            ridx[r['from_ngram']['text']] = list(set([r2['to_ngram']['text'] for r2 in rhymes.values()
                                                 if r2['from_ngram']['text'] == r['from_ngram']['text']]))

        nmulti = [n['text'] for n in tqdm(ngrams.values(), desc='prepping multi rhymes')
                  if (n.get('song_count', 0) > 2)
                  and (n['n'] in (2, 3,))
                  and (mscores_cache.get(n['text'], get_mscore) > 3)
                  and (not is_repeated(n['text']))]

        for ngram in tqdm(nmulti, desc='making multi rhymes'):
            grams = ngram.split()
            lists = []
            for gram in grams:
                rtos = ridx.get(gram, [])
                lists.append(rtos)
            combos = [p for p in product(*lists)]
            for c in combos:
                val = ' '.join(c)
                entry = ngrams.get(val)
                if (entry
                    and (entry.get('song_count', 0) > 2)
                    and (get_mscore(val) > 3)
                    and (not is_repeated(val))
                ):
                    rkey = (ngram, val, None)
                    if rkey not in rhymes:
                        rhymes[rkey] = dict(from_ngram=ngrams[ngram], to_ngram=ngrams[val], song=None, level=3)

        l1_rhymes = list(rhymes.values())
        for l1 in tqdm(l1_rhymes, desc='l2 rhymes'):
            for l2 in [r for r in l1_rhymes if r['from_ngram']['text'] == l1['to_ngram']['text']]:
                rkey = (l1['from_ngram']['text'], l2['to_ngram']['text'], None)
                if rkey not in rhymes:
                    rhymes[rkey] = dict(from_ngram=l1['from_ngram'], to_ngram=l2['to_ngram'], song=None, level=2)

        for ngram in tqdm(ngrams.values(), desc='ngram phones'):
            ngram['phones'] = phones_cache.get(ngram['text'], phones_getter)

        for ngram in tqdm(ngrams.values(), desc='ngram mscores'):
            ngram['mscore'] = mscores_cache.get(ngram['text'], get_mscore)

        ipa_cache, _ = Cache.objects.get_or_create(key='ngram_ipa')
        for ngram in tqdm(ngrams.values(), desc='ngram ipa'):
            ngram['ipa'] = ipa_cache.get(ngram['text'], get_ipa)

        n_counts = [0 for _ in range(15)]
        for sn in tqdm(song_ngrams.values(), desc='index ngram counts'):
            n = sn['ngram']['n']
            ct = sn['count']
            n_counts[n - 1] = (n_counts[n - 1] or 0) + ct
            ngram['count'] = (sn['ngram'].get('count') or 0) + ct
            ngrams[sn['ngram']['text']]['count'] = ngram['count']
        word_ct = n_counts[0]

        for ngram in tqdm(ngrams.values(), desc='score ngrams'):
            if ngram['n'] > 1:
                subgrams = [ngrams[gram] for gram in ngram['text'].split() if gram and (gram in ngrams)]
                total_with_same_n = n_counts[ngram['n'] - 1]
                ngram_pct = float((ngram.get('count') or 0.0) / total_with_same_n)
                chance_pct = 1.0
                for gram in subgrams:
                    gram_pct = float((gram.get('count') or 0.0) / word_ct)
                    chance_pct *= gram_pct
                ngram['pct'] = ngram_pct
                ngram['adj_pct'] = ngram_pct - chance_pct
            else:
                ngram['pct'] = float((ngram.get('count') or 0.0) / word_ct)
                ngram['adj_pct'] = float((ngram.get('count') or 0.0) / word_ct)

        print('ngrams', len(ngrams.values()))
        print('rhymes', len(rhymes.values()))
        print('song_ngrams', len(song_ngrams.values()))

        if dry_run:
            return

        print('saving db caches')
        datamuse_cache.save()
        phones_cache.save()
        mscores_cache.save()
        ipa_cache.save()

        del datamuse_cache, phones_cache, mscores_cache, ipa_cache, songs
        gc.collect()

        with transaction.atomic():
            print('deleting')
            Rhyme.objects.all().delete()
            NGram.objects.all().delete()
            SongNGram.objects.all().delete()

            ngrams = [NGram(**{k: v for k, v in n.items() if k not in ['count', 'song_count']})
                      for n in tqdm(ngrams.values(), desc='prepping ngrams')]
            print('writing ngrams', len(ngrams))
            ngrams = NGram.objects.bulk_create(ngrams, batch_size=batch_size)
            ngrams = dict([(n.text, n) for n in ngrams])

            rhyme_objs = []
            for rhyme in tqdm(rhymes.values(), desc='prepping rhymes'):
                nfrom = ngrams[rhyme['from_ngram']['text']]
                nto = ngrams[rhyme['to_ngram']['text']]
                song = rhyme['song']
                rhyme_objs.append(Rhyme(from_ngram=nfrom, to_ngram=nto, song=song, level=rhyme['level']))
                revkey = (nto.text, nfrom.text, song.id if song else None)
                if revkey not in rhymes:
                    rhyme_objs.append(Rhyme(from_ngram=nto, to_ngram=nfrom, song=song, level=rhyme['level']))
            print('writing rhymes', len(rhyme_objs))
            Rhyme.objects.bulk_create(rhyme_objs, batch_size=batch_size)
            del rhyme_objs
            gc.collect()

            sn_objs = []
            for sn in tqdm(song_ngrams.values(), desc='prepping song_ngrams'):
                n = ngrams[sn['ngram']['text']]
                sn_objs.append(SongNGram(song=sn['song'], ngram=n, count=sn['count']))
            print('creating song_ngrams', len(sn_objs))
            SongNGram.objects.bulk_create(sn_objs, batch_size=batch_size)

            if do_caches:
                reset_caches()

            print('finishing transaction')
        print('done')


def reset_caches():
    print('clearing caches')
    cc = cache._cache.get_client()
    keys = cc.keys('sism:*:suggest_*') + \
           cc.keys('sism:*:query_*') + \
           cc.keys('sism:*:top_*')
    for key in keys:
        key = str(key).split(':')[-1][:-1]
        cache.delete(key)

    topn = 200
    page_size = 50
    qsize = 200
    sug_size = 50

    Rhyme.objects.HARD_LIMIT = topn
    top = []
    for pg in tqdm(range(0, int(topn / page_size)), desc='top rhymes cache'):
        top += Rhyme.objects.top_rhymes(limit=page_size, offset=pg*page_size)

    for ngram in tqdm(top, desc='queries cache'):
        val = ngram['ngram']
        Rhyme.objects.query(q=val, limit=qsize)
        for i in range(0, len(val) + 1):
            qsug = val[:i]
            NGram.objects.suggest(qsug, sug_size)


def phones_getter(key):
    return get_phones(key, vowels_only=True, include_stresses=False)


def fetch_datamuse_rhymes(key):
    query = urlencode(dict(rel_rhy=key, max=50))
    query2 = urlencode(dict(rel_nry=key, max=50))
    vals = []
    try:
        vals += requests.get(f'https://api.datamuse.com/words?{query}').json()
    except:
        print('error retrieving datamuse RHY for', key)
    try:
        vals += requests.get(f'https://api.datamuse.com/words?{query2}').json()
    except:
        print('error retrieving datamuse NRY for', key)
    return vals


def is_repeated(w):
    return len(set(w.split())) < len(w.split())
