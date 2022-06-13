#!/usr/bin/env python

import os, django
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "songisms.settings")
django.setup()

from api.nlp_utils import tokenize_lyrics
from api.models import Song

docs = []
for song in Song.objects.exclude(lyrics=None):
    processed = tokenize_lyrics(song.lyrics)
    docs.append(TaggedDocument(processed, [song.title]))

print('training...')
model = Doc2Vec(vector_size=50, min_count=1, epochs=40, window=5)
model.build_vocab(docs)
model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)
model.save('./data/doc2vec.model')
