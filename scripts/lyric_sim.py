#!/usr/bin/env python ./manage.py script

import sys
import gensim.models
from pprint import pprint
from api.models import Song
from api.utils.text import *


def train():
    docs = []
    for song in Song.objects.exclude(lyrics=None).exclude(is_new=True):
        text = song.lyrics
        toks = [ tok for tok in tokenize_lyric(text) if get_mscore(tok) > 4 ]
        docs.append(toks)

    model = gensim.models.Word2Vec(docs)
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
