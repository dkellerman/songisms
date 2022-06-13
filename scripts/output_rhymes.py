#!/usr/bin/env python

import os, django, random
from tqdm import tqdm

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "songisms.settings")
django.setup()

from api.models import *

lines = set()
for s in tqdm(Song.objects.all()):
  if s.rhymes_raw:
    for l in s.rhymes_raw.split('\n'):
      if not l.strip():
        continue
      words = []
      for w in l.split(';'):
        w = '_'.join(w.split()).strip()
        words.append(w)
      line = ' '.join(words)
      lines.add(line)

lines = list(lines)
print('writing', len(lines))
with open('./data/rhymes.txt', 'w') as f:
  f.write('\n'.join(lines))


rhymes = list(Rhyme.objects.all().prefetch_related('to_ngram', 'from_ngram'))
random.shuffle(rhymes)
lines = set()
i = 0
while len(lines) < 2000:
    i += 1
    r = rhymes[i]
    anchor = r.from_ngram.text
    positive = r.to_ngram.text
    negative = rhymes[random.randint(0, len(rhymes))].to_ngram.text
    if anchor != positive:
        lines.add((anchor, positive, '1'))
        lines.add((anchor, negative, '0'))

with open('./data/rhymes_train.txt', 'w') as f:
    f.write('\n'.join([';'.join(l) for l in lines]))
