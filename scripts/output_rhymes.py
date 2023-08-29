#!/usr/bin/env python ./manage.py script

import random
import json
from tqdm import tqdm
from songs.models import Song
from rhymes.models import Rhyme


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


# output siamese neural net training data triples
rhymes = list(Rhyme.objects.filter(
    level=1).prefetch_related('to_ngram', 'from_ngram'))
random.shuffle(rhymes)
lines = set()
i = 0
while len(lines) < 5000:
    i += 1
    r = rhymes[i]
    anchor = r.from_ngram.text
    positive = r.to_ngram.text
    negative = rhymes[random.randint(0, len(rhymes) - 1)].to_ngram.text
    if anchor != negative:
        lines.add((anchor, positive, negative))

with open('./data/rhymes_train.txt', 'w') as f:
    f.write('\n'.join(';'.join(l) for l in lines))
