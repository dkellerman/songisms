#!/usr/bin/env python

import os, django, sys
from pprint import pprint
from gensim.models.doc2vec import Doc2Vec
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "songisms.settings")
django.setup()
from api.nlp_utils import tokenize_lyrics

if __name__ == '__main__':
    if len(sys.argv) > 1:
        line = sys.argv[1]
        print(line)
        model = Doc2Vec.load('./data/doc2vec.model')
        toks = tokenize_lyrics(line)
        vec = model.infer_vector(toks)
        pprint(model.dv.most_similar(vec))
    else:
        print("specify text")
