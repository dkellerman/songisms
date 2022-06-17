#!/usr/bin/env python

import sys, os, django
import gensim.models
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "songisms.settings")
django.setup()
from api.nlp_utils import *
from pprint import pprint


if __name__ == '__main__':
    if len(sys.argv) > 1:
        model = gensim.models.Word2Vec.load('./data/lyrics.w2v')
        line = sys.argv[1]
        pprint(model.wv.most_similar(line))
    else:
        print("specify text")
