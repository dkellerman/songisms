#!/usr/bin/env python ./manage.py script

import sys
import json
from tqdm import tqdm
from api.models import *
from api.utils.text import get_ipa_words, proc_text, score_rhyme


if __name__ == '__main__':
    args = sys.argv[3:]

    if len(args) == 1:
        if args[0] == 'bad':
            bad = []
            ngrams = NGram.objects.all()
            for n in tqdm(ngrams):
                ipa = get_ipa_words(n.text)
                if any([ not w.strip() or w == "'" or '*' in w for w in ipa ]):
                    bad.append(n.text)
            print(json.dumps(bad, indent=2))

        else:
            print("IPA", proc_text(args[0]))
            vals = []
            for n in tqdm(NGram.objects.all()):
                s = score_rhyme(args[0], n.text)
                vals.append((s, n.text))
            vals = sorted(vals, key=lambda x: x[0])
            print(vals[:50])

    elif len(args) == 2:
        s = score_rhyme(args[0], args[1])
        print(s)

    else:
        for r in tqdm(Rhyme.objects.all()):
            scores = []
            r1, r2 = r.from_ngram.text, r.to_ngram.text
            s = score_rhyme(r1, r2)
            scores.append(s)
            print('*', r1, r2, s)
        print('\n\n====== AVG:', sum(scores) / len(scores))

