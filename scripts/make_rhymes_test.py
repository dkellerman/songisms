#!/usr/bin/env python ./manage.py script

from tqdm import tqdm
from songs.models import Song
from rhymes.models import Vote
from songisms import utils


def make_vals():
    vals = set()
    for neg in tqdm(Vote.objects.filter(alt2=None, label='bad'), "Negatives"):
        vals.add((0.0, neg.anchor, neg.alt1, ''),)

    target_ct = len(vals) * 2
    prog_bar = tqdm(total=target_ct / 2, desc="Song rhymes")
    for s in Song.objects.filter(is_new=False).exclude(rhymes_raw=None):
        for line in s.rhymes_raw.split('\n'):
            line = line.strip()
            rsets = utils.get_rhyme_pairs(line)
            for anc, pos in rsets:
                if (0.0, anc, pos) in vals or (0.0, pos, anc) in vals:
                    continue
                vals.add((1.0, anc, pos, s.spotify_id))
                prog_bar.update()
                if len(vals) >= target_ct:
                    return vals


def output(vals):
    with open('./data/rhymes_test.csv', 'w') as f:
        vals = reversed(sorted(list(vals), key=lambda x: x[0]))
        f.write('anchor,other,score,song_id\n')
        f.write('\n'.join([','.join([str(score), anc, oth, song_id])
                          for score, anc, oth, song_id in vals]))


vals = make_vals()
output(vals)
