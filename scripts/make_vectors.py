#!/usr/bin/env python

import os, django
import gensim.models

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "songisms.settings")
django.setup()

from api.nlp_utils import *
from api.models import Song

docs = []
for song in Song.objects.exclude(lyrics=None):
    lines = song.lyrics.split('\n')
    doc = set()
    for line in lines:
        for tok in tokenize_lyric_line(line):
            doc.add(tok)
    docs.append(list(doc))

model = gensim.models.Word2Vec(docs)
model.save('./data/lyrics.w2v')
