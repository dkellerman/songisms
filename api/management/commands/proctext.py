#!/usr/bin/env python

from urllib.parse import urlencode
import requests_cache
from django.core.management.base import BaseCommand
from django.db import transaction
from api.models import *
from api.nlp_utils import *
from api.utils import JSONFileCache
from tqdm import tqdm


class Command(BaseCommand):
    help = 'Process text'

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        songs = Song.objects.all()
        ngrams = dict()
        rhymes = dict()
        song_ngrams = dict()

        # lyric ngrams
        print('lyric ngrams')
        for idx, song in enumerate(tqdm(songs)):
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

        print('extra ngrams')
        extra = []
        # with open('./data/mine.txt', 'r') as f:
        #     extra += get_lyric_ngrams(f.read(), range(5))
        with open('./data/idioms.txt', 'r') as f:
            extra += get_lyric_ngrams(f.read(), range(5))
        for text, n in tqdm(extra):
            ngrams[text] = ngrams.get(text, None) or dict(text=text, n=n)

        print('datamuse rhymes')
        datamuse_session = requests_cache.CachedSession('data/cache/datamuse_cache')
        n1 = [n for n in ngrams.values() if n['n'] == 1]
        common = get_common_words()
        rmuse = set()
        for from_ngram in tqdm(n1):
            query = urlencode(dict(rel_rhy=from_ngram['text'], max=50))
            query2 = urlencode(dict(rel_nry=from_ngram['text'], max=50))
            vals = []
            try:
                vals += datamuse_session.get(f'https://api.datamuse.com/words?{query}').json()
            except:
                print('error retrieving datamuse RHY for', from_ngram['text'])
            try:
                vals += datamuse_session.get(f'https://api.datamuse.com/words?{query2}').json()
            except:
                print('error retrieving datamuse NRY for', from_ngram['text'])
            for val in vals:
                to_word = val['word']
                from_word = from_ngram['text']
                if to_word in common and from_word in common:
                    ngrams[to_word] = ngrams.get(to_word) or dict(text=to_word, n=len(to_word.split()))
                    rkey = (from_ngram['text'], to_word, None)
                    if rkey not in rhymes:
                        rhymes[rkey] = dict(from_ngram=from_ngram, to_ngram=ngrams[to_word], song=None, level=1)
                        rmuse.add(to_word)

        print('l2 rhymes')
        l1_rhymes = list(rhymes.values())
        for l1 in tqdm(l1_rhymes):
            for l2 in [r for r in l1_rhymes if r['from_ngram']['text'] == l1['to_ngram']['text']]:
                rkey = (l1['from_ngram']['text'], l2['to_ngram']['text'], None)
                if rkey not in rhymes:
                    rhymes[rkey] = dict(from_ngram=l1['from_ngram'], to_ngram=l2['to_ngram'], song=None, level=2)

        print('ngram phones')
        phones_cache = JSONFileCache('./data/cache/phones.json', lambda key:
                                     get_phones(key, vowels_only=True, include_stresses=False))
        for ngram in tqdm(ngrams.values()):
            ngram['phones'] = phones_cache.get(ngram['text'])
        phones_cache.save()

        print('ngram mscore')
        mscores_cache = JSONFileCache('./data/cache/mscores.json', lambda key: get_mscore(key))
        for ngram in tqdm(ngrams.values()):
            ngram['mscore'] = mscores_cache.get(ngram['text'])
        mscores_cache.save()

        # print('ngram ipa')
        # ipa_cache = JSONFileCache('./data/cache/ngram_ipa.json', lambda key: get_ipa(key))
        # for ngram in tqdm(ngrams.values()):
        #     ngram['ipa'] = ipa_cache.get(ngram['text'])
        # ipa_cache.save()

        print('index ngram counts')
        n_counts = [0 for _ in range(15)]
        for sn in tqdm(song_ngrams.values()):
            n = sn['ngram']['n']
            ct = sn['count']
            n_counts[n - 1] = (n_counts[n - 1] or 0) + ct
            ngram['count'] = (sn['ngram'].get('count') or 0) + ct
            ngrams[sn['ngram']['text']]['count'] = ngram['count']
        word_ct = n_counts[0]

        print('score ngrams')
        for ngram in tqdm(ngrams.values()):
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

        with transaction.atomic():
            print('deleting')
            Rhyme.objects.all().delete()
            NGram.objects.all().delete()
            SongNGram.objects.all().delete()

            print('prepping ngrams')
            ngrams = [NGram(**{k: v for k, v in n.items() if k not in ['count', 'song_count']})
                      for n in tqdm(ngrams.values())]
            print('writing ngrams', len(ngrams))
            ngrams = NGram.objects.bulk_create(ngrams)
            ngrams = dict([(n.text, n) for n in ngrams])

            print('prepping rhymes')
            rhyme_objs = []
            for rhyme in tqdm(rhymes.values()):
                nfrom = ngrams[rhyme['from_ngram']['text']]
                nto = ngrams[rhyme['to_ngram']['text']]
                song = rhyme['song']
                rhyme_objs.append(Rhyme(from_ngram=nfrom, to_ngram=nto, song=song, level=rhyme['level']))
                revkey = (nto.text, nfrom.text, song.id if song else None)
                if revkey not in rhymes:
                    rhyme_objs.append(Rhyme(from_ngram=nto, to_ngram=nfrom, song=song, level=rhyme['level']))
            print('writing rhymes', len(rhyme_objs))
            Rhyme.objects.bulk_create(rhyme_objs)

            print('prepping song_ngrams')
            sn_objs = []
            for sn in tqdm(song_ngrams.values()):
                n = ngrams[sn['ngram']['text']]
                sn_objs.append(SongNGram(song=sn['song'], ngram=n, count=sn['count']))
            print('creating song_ngrams', len(sn_objs))
            SongNGram.objects.bulk_create(sn_objs)

            print('done')

