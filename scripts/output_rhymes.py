#!/usr/bin/env python ./manage.py script

import json
from tqdm import tqdm
from songs.models import Song

# output all rhyme groups to JSON
lines = list()
for s in tqdm(Song.objects.all()):
    if s.rhymes_raw:
        for l in s.rhymes_raw.split('\n'):
            if not l.strip():
                continue
            words = []
            for w in l.split(';'):
                words.append(w.strip())
            lines.append(words)

with open('./data/rhymes.json', 'w') as f:
    f.write(json.dumps(lines, indent=2))
