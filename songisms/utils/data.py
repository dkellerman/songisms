'''Wrappers for data files, with caching'''

import json
from functools import lru_cache


@lru_cache(maxsize=None)
def get_common_words(n=700):
    from nltk.corpus import brown
    from nltk import FreqDist
    fd = FreqDist(i.lower() for i in brown.words())
    return dict(fd.most_common()[:n])


@lru_cache(maxsize=None)
def get_custom_variants():
    with open('./data/variants.txt', 'r') as f:
        return [
            [l.strip() for l in line.split(';')]
            for line in f.readlines()
        ]


@lru_cache(maxsize=None)
def get_misspellings():
    with open('./data/misspellings.txt', 'r') as f:
        lines = [l.lower().strip().split('->')  for l in f.read().split('\n') if l.strip()]
        return { l[0]: l[1].split(',')[0].strip() for l in lines }


@lru_cache(maxsize=None)
def get_sim_sounds():
    with open('./data/simsounds.json', 'r') as f:
        return json.loads(f.read())


@lru_cache(maxsize=None)
def get_gpt_ipa():
    with open('./data/ipa_gpt.json', 'r') as f:
        return json.loads(f.read())


@lru_cache(maxsize=None)
def get_idioms():
    with open('./data/idioms.txt', 'r') as f:
        return [l.strip() for l in f.read().split('\n')]


@lru_cache(maxsize=None)
def get_mine():
    with open('./data/mine.txt', 'r') as f:
        return [l.strip() for l in f.read().split('\n')]
