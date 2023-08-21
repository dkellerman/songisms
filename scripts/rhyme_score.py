#!/usr/bin/env python ./manage.py script

import sys
import re
import g2p
import eng_to_ipa
import json
from scipy.spatial import distance
from tqdm import tqdm
from functools import lru_cache
from api.models import *
from panphon import featuretable
from minineedle import needle, core

ftable = featuretable.FeatureTable()
transducer = g2p.make_g2p('eng', 'eng-ipa')


@lru_cache(maxsize=None)
def get_ipa_words(text):
    global transducer
    words = eng_to_ipa.convert(text).split()
    ipa = [
        w if '*' not in w
        # TOOD: custom lookup as well?
        else get_g2p_word(w.replace('*', ''))
        for w in words
    ]
    ipa = [ fix_ipa_word(w) for w in ipa ]
    return ipa


def fix_ipa_word(w):
    if w is None:
        return ''
    w = re.sub(r"'ɛs$", "s", w)
    w = re.sub(r"'", "", w)
    return w.strip()


def remove_stresses(text):
    return re.sub(r'[\ˈˌ]', '', text)


def get_g2p_word(w):
    if w[-1] == "'":
        return re.sub(r'ŋ$', 'n', transducer(w[:-1] + 'g').output_string)
    return transducer(w).output_string


def get_ipa_features(ipa_letter):
    global ftable
    f = ftable.fts(ipa_letter)
    return f


def is_vowel(ipa_letter):
    return ipa_letter in ['ɪ', 'e', 'æ', 'ʌ', 'ʊ', 'ɒ', 'ə', 'i', 'ɑ', 'ɔ', 'ɜ', 'u', 'ɛ' ]


def get_stress_tail(phrase):
    if not phrase.strip():
        return ''
    stress_index = phrase.find("ˈ") + 1
    while not is_vowel(phrase[stress_index]):
        stress_index += 1
        if stress_index > len(phrase) - 1:
            return phrase[stress_index - 1:]
    return phrase[stress_index:]


@lru_cache(maxsize=None)
def align_ipa(ipa1, ipa2):
    if not ipa1.strip() or not ipa2.strip():
        return '', '', 0.0, None
    a = needle.NeedlemanWunsch(ipa1, ipa2)
    a.align()
    a1, a2 = a.get_aligned_sequences(core.AlignmentFormat.str)
    score = a.get_score()
    return a1, a2, score, a


def proc_text(text):
    text = text.lower().strip()
    ipa = get_ipa_words(text)
    text = ' '.join(ipa)
    tail = get_stress_tail(text)
    val = ''.join(tail.split(' '))
    return remove_stresses(val)


def score_rhyme(text1, text2):
    ipa1 = proc_text(text1)
    ipa2 = proc_text(text2)
    seq1, seq2, _, _ = align_ipa(ipa1, ipa2)
    f1, f2 = [], []
    for idx, c in enumerate(seq1):
        if c == '-' or seq2[idx] == '-' or c == '' or seq2[idx] == '':
            f1 += [0.0]
            f2 += [0.0]
        else:
            ft1 = get_ipa_features(c)
            ft2 = get_ipa_features(seq2[idx])
            if ft1 or ft2 is None:
                f1 += [0.0]
                f2 += [0.0]
            else:
                f1 += [ float(f) for f in ft1.numeric() ]
                f2 += [ float(f) for f in ft2.numeric() ]
    score = distance.cosine(f1, f2)
    return score


if __name__ == '__main__':
    args = sys.argv[3:]

    if len(args) == 1:
        if args[0] == 'check':
            bad = []
            ngrams = NGram.objects.filter(n=1)
            for n in tqdm(ngrams):
                ipa = get_ipa_words(n.text)
                if any([ not w.strip() or w == "'" or '*' in w for w in ipa ]):
                    bad.append(n.text)
            print(json.dumps(bad, indent=2))

        else:
            # print("IPA", proc_text(args[0]))
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

