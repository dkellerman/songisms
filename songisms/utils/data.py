'''Wrappers for data files, with caching
'''

import json
import csv
from functools import cached_property
from songisms import utils


class Data:
    @cached_property
    def common_words(self):
        return self.get_common_words()

    def get_common_words(self, n=500):
        from nltk.corpus import brown
        from nltk import FreqDist
        fd = FreqDist(utils.normalize_lyric(i)
                      for i in brown.words()
                      if utils.remove_all_punctuation(i).strip())
        return dict(fd.most_common()[:n])

    @cached_property
    def custom_variants(self):
        return csv.reader(open('./data/variants.csv', 'r'))

    @cached_property
    def misspellings(self):
        with open('./data/misspellings.csv', 'r') as f:
            return {vals[0]: vals[1:] for vals in csv.reader(f)}

    @cached_property
    def sim_sounds(self):
        with open('./data/simsounds.json', 'r') as f:
            return json.loads(f.read())

    @cached_property
    def idioms(self):
        return [l[0].strip().lower() for l in csv.reader(open('./data/idioms.csv', 'r'))]

    @cached_property
    def lines(self):
        return [l[0].strip().lower() for l in csv.reader(open('./data/lines.csv', 'r'))]

    @cached_property
    def rhymes_train(self):
        with open('./data/rhymes_train.csv', 'r') as f:
            all = []
            for vals in csv.reader(f):
                score = vals[0]
                rhymes = utils.get_rhyme_pairs(';'.join(vals[1:]))
                for r1, r2 in rhymes:
                    all.append((float(score), r1, r2))
            return all


    @cached_property
    def rhymes_test(self):
        with open('./data/rhymes_test.csv', 'r') as f:
            all = []
            for score, anc, oth, _ in list(csv.reader(f))[1:]:
                all.append((float(score), anc, oth))
            return all

data = Data()
