#!/usr/bin/env python

import os, django
from tqdm import tqdm

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "songisms.settings")
django.setup()

from api.models import Song

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
with open('rhymeslist.txt', 'w') as f:
  f.write('\n'.join(lines))

