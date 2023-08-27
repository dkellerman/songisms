#!/usr/bin/env python ./manage.py script

import numpy as np
import sys
import gensim.models
from pprint import pprint
from api.models import Song
from api.utils.text import *
from gensim.models import Word2Vec
import gensim.downloader


def train():
    print("Downloading...")
    kv = gensim.downloader.load("glove-twitter-25")

    print("Building model...")
    model = Word2Vec(vector_size=kv.vector_size, window=5, min_count=1, workers=4)
    model.build_vocab([list(kv.index_to_key)], update=False)
    for word in kv.index_to_key:
        if word in model.wv.index_to_key:
            model.wv[word] = kv[word]

    print("Preparing data...")
    docs = []
    for song in Song.objects.exclude(lyrics=None).exclude(is_new=True):
        text = song.lyrics
        toks = [ tok for tok in tokenize_lyric(text) if get_mscore(tok) > 4 ]
        docs.append(toks)
    model.build_vocab(docs, update=True)

    print("Training...")
    model.train(docs, total_examples=len(docs), epochs=3)
    print("Saving...")
    model.save('./data/lyrics.w2v')


def sim(q):
    model = gensim.models.Word2Vec.load('./data/lyrics.w2v')
    pprint(model.wv.most_similar(q))


if __name__ == '__main__':
    args = sys.argv[3:]
    if len(args) == 0:
        print("Training...")
        train()
    else:
        sim(args[0])
