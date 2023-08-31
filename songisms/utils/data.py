'''Wrappers for data files, with caching
'''

import json
from functools import cached_property

class Data:
    @cached_property
    def common_words(self):
        from nltk.corpus import brown
        from nltk import FreqDist
        fd = FreqDist(i.lower() for i in brown.words())
        return dict(fd.most_common()[:1000])

    @cached_property
    def custom_variants(self):
        with open('./data/variants.txt', 'r') as f:
            return [
                [l.strip() for l in line.split(';')]
                for line in f.readlines()
            ]

    @cached_property
    def misspellings(self):
        with open('./data/misspellings.txt', 'r') as f:
            lines = [l.lower().strip().split('->')  for l in f.read().split('\n') if l.strip()]
            return { l[0]: l[1].split(',')[0].strip() for l in lines }

    @cached_property
    def sim_sounds(self):
        with open('./data/simsounds.json', 'r') as f:
            return json.loads(f.read())

    @cached_property
    def gpt_ipa(self):
        with open('./data/ipa_gpt.json', 'r') as f:
            return json.loads(f.read())

    @cached_property
    def idioms(self):
        with open('./data/idioms.txt', 'r') as f:
            return [l.strip() for l in f.read().split('\n')]

    @cached_property
    def mine(self):
        with open('./data/mine.txt', 'r') as f:
            return [l.strip() for l in f.read().split('\n')]

    @cached_property
    def rhymes(self):
        with open('./data/rhymes.json', 'r') as f:
            return json.loads(f.read())

    @cached_property
    def rhymes_train(self):
        with open('./data/rhymes_train.csv', 'r') as f:
            return [l.strip().split(';') for l in f.read().split('\n')]


data = Data()
