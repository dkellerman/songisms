#!/usr/bin/env python ./manage.py script

import sys
import re
import g2p
import eng_to_ipa
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
    return ftable.fts(ipa_letter)


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
    ipa = get_ipa_words(text)
    text = ' '.join(ipa)
    tail = get_stress_tail(text)
    val = ''.join(tail.split(' '))
    return remove_stresses(val)


if __name__ == '__main__':
    args = sys.argv[3:]
    if len(args) == 1:
        print("IPA", proc_text(args[0]))

    elif len(args) == 2:
        ipa1 = proc_text(args[0])
        ipa2 = proc_text(args[1])
        seq1, seq2, score, _ = align_ipa(ipa1, ipa2)
        print(seq1)
        print(seq2)
        print(score)

    else:
        for r in tqdm(Rhyme.objects.all()):
            scores = []
            r1 = proc_text(r.from_ngram.text)
            r2 = proc_text(r.to_ngram.text)
            seq1, seq2, score, _ = align_ipa(r1, r2)
            scores.append(score)
        print(sum(scores) / len(scores))
