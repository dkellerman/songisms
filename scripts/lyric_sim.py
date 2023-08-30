#!/usr/bin/env python ./manage.py script

import sys
import multiprocessing as mp
from pprint import pprint
from songs.models import Song
from songisms import utils
import gensim.models
import gensim.downloader
from gensim.models import Word2Vec


def train():
    print("Downloading...")
    kv = gensim.downloader.load("glove-twitter-25")

    print("Building model...")
    cores = mp.cpu_count()
    model = Word2Vec(vector_size=kv.vector_size, window=5, min_count=1, workers=cores)
    model.build_vocab([list(kv.index_to_key)], update=False)
    for word in kv.index_to_key:
        if word in model.wv.index_to_key:
            model.wv[word] = kv[word]

    print("Preparing data...")
    docs = []
    for song in Song.objects.exclude(lyrics=None).exclude(is_new=True):
        text = song.lyrics
        for line in text.split('\n'):
            if line.strip() == '':
                continue
            toks = [tok for tok in utils.tokenize_lyric(line) if utils.get_mscore(tok) > 3]
            docs.append(toks)
    model.build_vocab(docs, update=True)

    print("Training...", len(docs))
    model.train(docs, total_examples=len(docs), epochs=1)

    print("Saving...")
    model.save('./data/lyrics.w2v')


def sim(q):
    model1 = gensim.downloader.load("glove-twitter-25")
    model2 = gensim.models.Word2Vec.load('./data/lyrics.w2v')
    print("Glove:")
    pprint(model1.most_similar(q))
    print("\n\nTrained:")
    pprint(model2.wv.most_similar(q))


if __name__ == '__main__':
    args = sys.argv[3:]
    if len(args) == 0:
        print("Training...")
        train()
    else:
        sim(args[0])
