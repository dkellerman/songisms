#!/usr/bin/env python

import json
from urllib.parse import urlencode
import requests_cache
from django.db import transaction
from django.core.management.base import BaseCommand
from api.models import *
from api.nlp_utils import *
from tqdm import tqdm

nlp = load_nlp()

_phones = None
_mscores = None
_ipa = None

_CACHES = dict(
    datamuse='data/cache/datamuse_cache',
    phones='./data/cache/phones.json',
    mscores='./data/cache/mscores.json',
    ipa='./data/cache/ngram_ipa.json',
)

class Command(BaseCommand):
    help = 'Process text'

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        songs = Song.objects.all()
        ngrams = dict()
        rhymes = list()
        song_ngrams = dict()
        session = requests_cache.CachedSession(_CACHES['datamuse'])

        # lyric ngrams
        print('lyric ngrams')
        for idx, song in enumerate(tqdm(songs)):
            if song.lyrics:
                texts = get_lyric_ngrams(song.lyrics, range(5))
                for text, n in texts:
                    ngrams[text] = ngrams.get(text, None) or dict(text=text, n=n)
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
                    rhymes.append(dict(from_ngram=ngrams[from_text], to_ngram=ngrams[to_text], song=song, level=1))

        print('extra ngrams')
        extra = []
        with open('./data/mine.txt', 'r') as f:
            extra += get_lyric_ngrams(f.read(), range(5))
        with open('./data/idioms.txt', 'r') as f:
            extra += get_lyric_ngrams(f.read(), range(5))
        for text, n in tqdm(extra):
            ngrams[text] = ngrams.get(text, None) or dict(text=text, n=n)

        print('datamuse rhymes')
        n1 = [n for n in ngrams.values() if n['n'] == 1]
        for from_ngram in tqdm(n1):
            query = urlencode(dict(rel_rhy=from_ngram['text'], max=50))
            query2 = urlencode(dict(rel_nry=from_ngram['text'], max=50))
            vals = []
            try:
                vals += session.get(f'https://api.datamuse.com/words?{query}').json()
            except:
                print('error retrieving datamuse RHY for', from_ngram['text'])
            try:
                vals += session.get(f'https://api.datamuse.com/words?{query2}').json()
            except:
                print('error retrieving datamuse NRY for', from_ngram['text'])
            for val in vals:
                to_word = val['word']
                if val in get_common_words():
                    ngrams[to_word] = ngrams.get(to_word) or NGram(text=to_word, n=len(to_word.split()))
                    rhymes.append(dict(from_ngram=from_ngram, to_ngram=ngrams[to_word], song=None, level=1))

        print('l2 rhymes')
        l1_rhymes = list(rhymes)
        for l1 in tqdm(l1_rhymes):
            for l2 in [r for r in l1_rhymes if r['from_ngram']['text'] == l1['to_ngram']['text']]:
                rhymes.append(dict(from_ngram=l1['from_ngram'], to_ngram=l2['to_ngram'], song=None, level=2))

        # meta
        global _phones, _mscores, _ipa

        print('ngram phones')
        for ngram in tqdm(ngrams.values()):
            ngram['phones'] = get_phones_for_text(ngram['text'])
        with open(_CACHES['phones'], 'w') as f:
            f.write(json.dumps(_phones))

        print('ngram mscore')
        for ngram in tqdm(ngrams.values()):
            ngram['mscore'] = get_mscore(ngram['text'])
        with open(_CACHES['mscores'], 'w') as f:
            f.write(json.dumps(_mscores))

        # print('ngram ipa')
        # for ngram in tqdm(ngrams.values()):
        #     ngram['ipa'] = get_ngram_ipa(ngram['text'])
        # with open(_CACHES['ipa'], 'w') as f:
        #     f.write(json.dumps(_ipa))

        print('index ngram counts')
        n_counts = [0 for _ in range(15)]
        for sn in tqdm(song_ngrams.values()):
            n = sn['ngram']['n']
            ct = sn['count']
            n_counts[n - 1] = (n_counts[n - 1] or 0) + ct
            ngram['count'] = (sn['ngram'].get('count') or 0) + ct
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
        print('rhymes', len(rhymes))
        print('song_ngrams', len(song_ngrams.values()))


        with transaction.atomic():
            print('deleting')
            Rhyme.objects.all().delete()
            NGram.objects.all().delete()
            SongNGram.objects.all().delete()

            print('prepping ngrams')
            ngrams = [NGram(**{k: v for k, v in n.items() if k != 'count'})
                      for n in tqdm(ngrams.values())]
            print('writing ngrams')
            ngrams = NGram.objects.bulk_create(ngrams)
            ngrams = dict([(n.text, n) for n in ngrams])

            print('prepping rhymes')
            rhyme_objs = []
            for rhyme in tqdm(rhymes):
                nfrom = ngrams[rhyme['from_ngram']['text']]
                nto = ngrams[rhyme['to_ngram']['text']]
                rhyme_objs.append(Rhyme(from_ngram=nfrom, to_ngram=nto, song=rhyme['song'], level=rhyme['level']))
            print('writing rhymes')
            Rhyme.objects.bulk_create(rhyme_objs)

            print('prepping song_ngrams')
            sn_objs = []
            for sn in tqdm(song_ngrams.values()):
                n = ngrams[sn['ngram']['text']]
                sn_objs.append(SongNGram(song=sn['song'], ngram=n, count=sn['count']))
            print('creating song_ngrams')
            SongNGram.objects.bulk_create(sn_objs)

            print('done')


def get_mscore(text):
    global _mscores
    if _mscores is None:
        try:
            with open(_CACHES['mscores'], 'r') as f:
                _mscores = json.loads(f.read())
        except:
            _mscores = {}
    mscore = _mscores.get(text, None)
    if mscore:
        return mscore
    mscore = [POS_TO_MSCORE.get(tok.pos_, 0) for tok in nlp(text)]
    mscore[-1] *= 1.5
    mscore = sum(mscore) / len(mscore)
    _mscores[text] = mscore
    return mscore


def get_phones_for_text(text):
    global _phones
    if _phones is None:
        try:
            with open(_CACHES['phones'], 'r') as f:
                _phones = json.loads(f.read())
        except:
            _phones = {}
    phones = _phones.get(text, None)
    if phones:
        return phones
    phones = get_phones(text, vowels_only=True, include_stresses=False)
    _phones[text] = phones
    return phones


def get_ngram_ipa(text):
    global _ipa
    if _ipa is None:
        try:
            with open(_CACHES['ipa'], 'r') as f:
                _ipa = json.loads(f.read())
        except:
            _ipa = {}
    ipa = _ipa.get(text, None)
    if ipa:
        return ipa
    ipa = get_ipa(text)
    _ipa[text] = ipa
    return ipa
