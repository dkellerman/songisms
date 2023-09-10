#!/usr/bin/env python ./manage.py script

import json
from tqdm import tqdm
from songs.models import Song
from rhymes.models import Vote
from songisms import utils


# votes
vals = set()

for neg in tqdm(Vote.objects.filter(alt2=None, label='bad'), "Negatives"):
    vals.add((0.0, neg.anchor, neg.alt1, ''),)


for s in tqdm(Song.objects.filter(is_new=False).exclude(rhymes_raw=None), "Song rhymes"):
    for line in s.rhymes_raw.split('\n'):
        line = line.strip()
        rsets = utils.get_rhyme_pairs(line)
        for anc, pos in rsets:
            if (0.0, anc, pos) in vals or (0.0, pos, anc) in vals:
                continue
            vals.add((1.0, anc, pos, s.spotify_id))


with open('./data/rhymes_test.csv', 'w') as f:
    f.write('anchor,other,score,song_id\n')
    f.write('\n'.join([','.join([str(score), anc, oth, song_id])
                       for score, anc, oth, song_id in vals]))
