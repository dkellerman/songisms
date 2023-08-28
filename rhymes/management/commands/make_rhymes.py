#!/usr/bin/env python

import argparse
import itertools
import gc
from itertools import product
from django.core.management.base import BaseCommand
from django.db import transaction
from nltk import FreqDist
from rhymes.models import NGram, Rhyme, SongNGram, Cache
from songisms.utils import (get_vowel_vector, get_lyric_ngrams, get_rhyme_pairs, get_common_words,
                            get_mscore, get_ipa_text, get_stresses_vector, fetch_datamuse_rhymes)
from tqdm import tqdm


class Command(BaseCommand):
    help = 'Process ngrams and rhymes'

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', '-D', action=argparse.BooleanOptionalAction)
        parser.add_argument('--batch-size', '-b', type=int, default=1000)

    def handle(self, *args, **options):
        try:
            from songs.models import Song  # songs app needs to be installed
        except:
            print('songs app not installed')
            return

        batch_size, dry_run = [options[k] for k in ('batch_size', 'dry_run',)]
        dry_run = options['dry_run']

        songs = Song.objects.filter(is_new=False).exclude(lyrics=None)
        ngrams = dict()
        rhymes = dict()
        song_ngrams = dict()

        vector_cache, _ = Cache.objects.get_or_create(key='ngram_vector')
        datamuse_cache, _ = Cache.objects.get_or_create(key='datamuse')

        for idx, song in enumerate(tqdm(songs, desc='lyric ngrams')):
            # lyric ngrams
            texts = get_lyric_ngrams(song.lyrics, range(5))
            for text, n in texts:
                ngrams[text] = ngrams.get(text, None) or dict(text=text, n=n)
                song_ngram = song_ngrams.get((song.spotify_id, text), None)
                if song_ngram:
                    song_ngram['count'] += 1
                else:
                    song_ngrams[(song.spotify_id, text)] = dict(ngram=ngrams[text], song=song.spotify_id, count=1)

            for text, _ in list(set(texts)):
                ngrams[text]['song_count'] = ngrams[text].get('song_count', 0) + 1

            if song.rhymes_raw:
                rhyme_pairs = get_rhyme_pairs(song.rhymes_raw)
                for from_text, to_text in rhyme_pairs:
                    ngrams[from_text] = ngrams.get(from_text) or dict(text=from_text, n=len(from_text.split()))
                    ngrams[to_text] = ngrams.get(to_text) or dict(text=to_text, n=len(to_text.split()))
                    rhymes[(from_text, to_text, song.spotify_id)] = \
                        dict(from_ngram=ngrams[from_text], to_ngram=ngrams[to_text], song=song.spotify_id, level=1)

        extra = []
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
                  and (get_mscore(n['text']) > 3)
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

        for ngram in tqdm(ngrams.values(), desc='ngram vectors'):
            ngram['phones'] = vector_cache.get(ngram['text'], vector_getter) or None

        for ngram in tqdm(ngrams.values(), desc='ngram mscores'):
            ngram['mscore'] = get_mscore(ngram['text'])

        ipa_cache, _ = Cache.objects.get_or_create(key='ngram_ipa')
        for ngram in tqdm(ngrams.values(), desc='ngram ipa'):
            ngram['ipa'] = ipa_cache.get(ngram['text'], get_ipa_text)

        stresses_cache, _ = Cache.objects.get_or_create(key='ngram_stresses')
        for ngram in tqdm(ngrams.values(), desc='ngram stresses'):
            ngram['stresses'] = stresses_cache.get(ngram['text'], get_stresses_vector)

        n_counts = [0 for _ in range(15)]
        for sn in tqdm(song_ngrams.values(), desc='index ngram counts'):
            n = sn['ngram']['n']
            ct = sn['count']
            n_counts[n - 1] = (n_counts[n - 1] or 0) + ct
            ngram['count'] = (sn['ngram'].get('count') or 0) + ct
            ngrams[sn['ngram']['text']]['count'] = ngram['count']
        word_ct = n_counts[0]

        song_ct = songs.count()
        title_ngrams = FreqDist(itertools.chain(*[n[0] for n in [get_lyric_ngrams(s.title) for s in songs]]))
        for ngram in tqdm(ngrams.values(), desc='ngrams pct'):
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
            ngram['title_pct'] = title_ngrams.freq(ngram['text'])
            ngram['song_pct'] = ngram.get('song_count', 0) / song_ct

        print('ngrams', len(ngrams.values()))
        print('rhymes', len(rhymes.values()))
        print('song_ngrams', len(song_ngrams.values()))

        if dry_run:
            return

        for c in tqdm([datamuse_cache, vector_cache, ipa_cache, stresses_cache], desc='saving db caches'):
            c.save()
            del c
        del title_ngrams
        gc.collect()

        with transaction.atomic():
            print('deleting')
            Rhyme.objects.all().delete()
            NGram.objects.all().delete()
            SongNGram.objects.all().delete()

            ngrams = [NGram(**{k: v for k, v in n.items()})
                      for n in tqdm(ngrams.values(), desc='prepping ngrams')]
            print('writing ngrams', len(ngrams))
            ngrams = NGram.objects.bulk_create(ngrams, batch_size=batch_size)
            ngrams = dict([(n.text, n) for n in ngrams])

            rhyme_objs = []
            for rhyme in tqdm(rhymes.values(), desc='prepping rhymes'):
                nfrom = ngrams[rhyme['from_ngram']['text']]
                nto = ngrams[rhyme['to_ngram']['text']]
                song = rhyme['song']
                rhyme_objs.append(Rhyme(from_ngram=nfrom, to_ngram=nto, song_uid=song, level=rhyme['level']))
                revkey = (nto.text, nfrom.text, song if song else None)
                if revkey not in rhymes:
                    rhyme_objs.append(Rhyme(from_ngram=nto, to_ngram=nfrom, song_uid=song, level=rhyme['level']))

            print('writing rhymes', len(rhyme_objs))
            Rhyme.objects.bulk_create(rhyme_objs, batch_size=batch_size)
            del rhyme_objs
            gc.collect()

            sn_objs = []
            for sn in tqdm(song_ngrams.values(), desc='prepping song_ngrams'):
                n = ngrams[sn['ngram']['text']]
                sn_objs.append(SongNGram(song_uid=sn['song'], ngram=n, count=sn['count']))
            print('creating song_ngrams', len(sn_objs))
            SongNGram.objects.bulk_create(sn_objs, batch_size=batch_size)

            print('finishing transaction')
        print('done')


def vector_getter(key):
    return get_vowel_vector(key)


def is_repeated(w):
    return len(set(w.split())) < len(w.split())