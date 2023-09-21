'''IPA and other pronounciation-related utilities
'''

import re
from functools import lru_cache
from collections import defaultdict
from typing import Any
from . import utils

espeak = None


@lru_cache(maxsize=None)
def get_ipa_cache():
    from rhymes.models import Cache
    cache, _ = Cache.objects.get_or_create(key='ipa')
    return cache


@lru_cache(maxsize=None)
def ftable():
    from panphon import featuretable
    return featuretable.FeatureTable()


def to_ipa_tokens(text):
    toks = utils.tokenize_lyric(text)
    ipa_toks = []
    for tok in toks:
        ipa = get_ipa_cache().get(tok, get_espeak_ipa)
        ipa_toks.append(ipa)
    return ipa_toks


def to_ipa(text, save_cache=True):
    text = utils.normalize_lyric(text)
    return get_ipa_cache().get(
        text,
        lambda t: ' '.join(to_ipa_tokens(t)),
        save=save_cache)


def get_espeak_ipa(text):
    global espeak
    if not espeak:
        from espeakng import ESpeakNG
        espeak = ESpeakNG()
        espeak.voice = 'en-us'
    return espeak.g2p(text, ipa=2)


def remove_stresses(text):
    return re.sub(r'\ˈ|\ˌ', '', text)


def get_ipa_features(ipa_letter) -> Any:
    '''Get feature table for IPA character
    '''
    f = ftable().fts(ipa_letter)
    return f


def get_ipa_features_vector(ipa):
    if type(ipa) == str:
        chars = [c for c in ipa]
    else:
        chars = ipa

    vec = []
    for char in chars:
        if char in ['_', ' ', '.']:
            features = [0.0] + utils.EMPTY_IPA_FEATURES
        elif char == "ˌ":
            features = [1.0] + utils.EMPTY_IPA_FEATURES
        elif char == "ˈ":
            features = [2.0] + utils.EMPTY_IPA_FEATURES
        elif char == "ː":
            features = [3.0] + utils.EMPTY_IPA_FEATURES
        elif char == "\u0361":
            features = [4.0] + utils.EMPTY_IPA_FEATURES
        else:
            ft = utils.get_ipa_features(char)
            if ft is None:
                features = [0.0] + utils.EMPTY_IPA_FEATURES
            else:
                features = [0.0] + ft.numeric()
        vec.append(features)

    return vec


def is_vowel(ipa_letter):
    return ipa_letter in IPA_VOWELS


def get_ipa_stress_tail(ipa_phrase):
    '''Stress tail is basically the primary stressed vowel and everything after, so
       include stress markers in the input string if possible.
    '''
    if not ipa_phrase.strip():
        return ''

    stress_index = ipa_phrase.find("ˈ") + 1
    while not is_vowel(ipa_phrase[stress_index]):
        stress_index += 1
        if stress_index > len(ipa_phrase) - 1:
            return ipa_phrase[stress_index - 1:]
    return ipa_phrase[stress_index:]


def chop_ipa_tail(ipa_phrase):
    if ipa_phrase[-1] == "ŋ":
        return ipa_phrase[:-1] + "n"
    index = len(ipa_phrase) - 1
    while not is_vowel(ipa_phrase[index]) and not ipa_phrase[index] == "n":
        index -= 1
        if index <= 0:
            return ipa_phrase
    return ipa_phrase[:index + 1]


FULL_IPA_FEATURE_LEN = len(get_ipa_features('a').numeric())

EMPTY_IPA_FEATURES = [0.0] * FULL_IPA_FEATURE_LEN

IPA_VOWELS = [u'i', u'y', u'e', u'ø', u'ɛ', u'œ', u'a', u'ɶ', u'ɑ', u'ɒ', u'ɔ',
              u'ʌ', u'ɤ', u'o', u'ɯ', u'u', u'ɪ', u'ʊ', u'ə', u'æ']

IPA_DIPTHONGS = ['eɪ', 'oʊ', 'aʊ', 'ɪə', 'eə', 'ɔɪ', 'aɪ', 'ʊə' ]


ipa_token_dict = defaultdict(lambda: len(ipa_token_dict))

def ipa_to_unique_tokens(text):
    return [ipa_token_dict[c] for c in text]


def is_valid_onset(onset):
    # https://en.wikipedia.org/wiki/English_phonology#Syllable_structure
    return (
        (onset != "ŋ")  # All single-consonant phonemes except /ŋ/

        or onset in [
            # Stop plus approximant other than /j/
            'pl', 'bl', 'kl', 'ɡl', 'pr', 'br', 'tr', 'dr', 'kr',
            'ɡr', 'tw', 'dw', 'ɡw', 'kw', 'pw',
            # Voiceless fricative or /v/ plus approximant other than /j/
            'fl', 'sl', 'θl', 'ʃl', 'fr', 'θr', 'ʃr', 'hw', 'sw', 'θw', 'vw',
            # Consonant other than /r/ or /w/ plus /j/ (before /uː/ or its modified/reduced forms)
            'pj', 'bj', 'tj', 'dj', 'kj', 'ɡj', 'mj', 'nj', 'fj', 'vj', 'θj',
            'sj', 'zj', 'hj', 'lj',
            # /s/ plus voiceless stop
            'sp', 'st', 'sk',
            # /s/ plus nasal other than /ŋ/
            'sm', 'sn',
            # /s/ plus voiceless non-sibilant fricative
            'sf', 'sθ',
            # /s/ plus voiceless stop plus approximant
            'spl', 'skl', 'spr', 'str', 'skr', 'skw', 'spj', 'stj', 'skj',
            # /s/ plus nasal plus approximant
            'smj',
            # /s/ plus voiceless non-sibilant fricative plus approximant
            'sfr',
            # ???
        ]
    )


def get_syllables_from_ipa(ipa, as_str=False, spaces=False):
    syllables = []
    ipa = [c for c in ipa]

    while len(ipa):  # per-word
        char = None
        onset = []
        coda = []
        nucleus = []

        while len(ipa) and char != ' ':  # per-syllable
            char = ipa.pop()

            if onset and not is_valid_onset(char + ''.join(onset)):
                print('invalid', char + ''.join(onset))
                ipa.append(char)
                ipa += onset
                onset = []
                break
            elif nucleus:
                if (char in IPA_VOWELS or char in ["ŋ"]) \
                    and (char + (''.join(nucleus)) not in IPA_DIPTHONGS
                ):
                    ipa.append(char)
                    break
                else:
                    onset.insert(0, char)
            else:
                if char in IPA_VOWELS:
                    nucleus = [char]
                else:
                    coda.insert(0, char)

        syl = [''.join(onset), ''.join(nucleus), ''.join(coda)]
        if not spaces:
            syl = [s.strip() for s in syl]

        syllables.insert(0, syl)

    if as_str:
        return '.'.join([''.join(syl) for syl in syllables])

    return syllables


def to_syllabified_ipa(text, as_str=True, spaces=False):
    ipa = to_ipa(text)
    return get_syllables_from_ipa(ipa, as_str=as_str, spaces=spaces)
