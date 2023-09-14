'''Wrappers for data files, with caching
'''

import json
from functools import cached_property
from songisms import utils


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
                [l.strip() for l in line.split(';') if l.strip()]
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
        vals = dict()
        with open('./data/ipa_gpt.json', 'r') as f:
            vals.update(json.loads(f.read()))
        with open('./data/ipa_gpt_custom.json', 'r') as f:
            vals.update(json.loads(f.read()))
        return vals

    @cached_property
    def idioms(self):
        with open('./data/idioms.txt', 'r') as f:
            return [l.strip() for l in f.read().split('\n') if l.strip()]

    @cached_property
    def lines(self):
        with open('./data/lines.txt', 'r') as f:
            return [l.strip() for l in f.read().split('\n') if l.strip()]

    @cached_property
    def rhymes_train(self):
        with open('./data/rhymes_train.csv', 'r') as f:
            all = []
            for line in f.read().split('\n'):
                line = line.strip()
                if not line:
                    continue
                vals = line.split(',')
                score = vals[0]
                rhymes = utils.get_rhyme_pairs(';'.join(vals[1:]))
                for r1, r2 in rhymes:
                    all.append((float(score), r1, r2))
            return all


    @cached_property
    def rhymes_test(self):
        with open('./data/rhymes_test.csv', 'r') as f:
            all = []
            for line in f.read().split('\n')[1:]:
                line = line.strip()
                if not line:
                    continue
                score, anc, oth, _ = line.split(',')
                all.append((float(score), anc, oth))
            return all


data = Data()
