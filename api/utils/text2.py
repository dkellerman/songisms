import re
import g2p
import eng_to_ipa
import string
import json
from scipy.spatial import distance
from functools import lru_cache
from panphon import featuretable
from minineedle import needle, core

ftable = featuretable.FeatureTable()
transducer = g2p.make_g2p('eng', 'eng-ipa')


def tokenize_lyric(val):
    val = val.lower().strip()
    val = re.sub(r'[\,\"]', '', val)
    val = re.sub(r'[\,\"]', '', val)
    val = re.sub(r'\)', '', val)
    val = re.sub(r'\(', '', val)
    val = re.sub(r'\.+(\s|$)', ' ', val)
    val = re.sub(r'[^\w\s\'-.]', '', val)
    val = re.sub(r'(\w+in)\'[^\w]', r'\1g ', val)
    val = re.sub(r'\s+', ' ', val)
    toks = [t for t in val.split() if t not in string.punctuation]
    toks = [t + ('.' if '.' in t else '') for t in toks]
    toks = [re.sub(r'\.+', '.', t) for t in toks]
    return toks


def normalize_lyric(val):
    return ' '.join(tokenize_lyric(val))


def normalize_ipa(ipa):
    ipa = ipa.strip().replace("ː", "")
    return remove_non_lyric_punctuation(ipa)


@lru_cache(maxsize=None)
def get_gpt_ipa():
    with open('./data/ipa_gpt.json', 'r') as f:
        return json.load(f)


@lru_cache(maxsize=None)
def get_ipa_words(text):
    global transducer
    words = eng_to_ipa.convert(text).split()
    gpt_ipa = get_gpt_ipa()
    ipa = []
    for w in words:
        if '*' in w or not w.strip():
            w = w.replace('*', '').strip()
            w = gpt_ipa.get(w, get_g2p_word(w))
        ipa.append(fix_ipa_word(w))
    return ipa


def fix_ipa_word(w):
    if w is None:
        return ''
    w = re.sub(r"'ɛs$", "s", w)
    w = re.sub(r"'", "", w)
    return w.strip()


def remove_stresses(text):
    return re.sub(r'\ˈ|\ˌ', '', text)


def remove_punctuation(text):
    return ''.join([t for t in text if t not in string.punctuation])


def remove_non_lyric_punctuation(text):
    return ''.join([t for t in text if t not in r"""!"#$%&()*+,./:;<=>?@[\]^_`{|}~"""])


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


def align_vals(val1, val2):
    a = needle.NeedlemanWunsch(val1, val2)
    a.gap_character = '_'
    a.align()
    fmt = core.AlignmentFormat.list if type(val1) == list else core.AlignmentFormat.str
    a1, a2 = a.get_aligned_sequences(fmt)
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
    seq1, seq2, _, _ = align_vals(ipa1, ipa2)
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
