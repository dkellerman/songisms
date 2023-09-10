'''IPA and other pronounciation-related utilities
'''

import re
from functools import lru_cache, cached_property
from collections import defaultdict
from typing import Any
from . import utils


@lru_cache(maxsize=None)
def ftable():
    from panphon import featuretable
    return featuretable.FeatureTable()


@lru_cache(maxsize=None)
def transducer():
    import g2p
    return g2p.make_g2p('eng', 'eng-ipa')


def normalize_ipa(ipa):
    '''Normalize IPA string
    '''
    if ipa.endswith("'") or ipa.endswith("ˌ"):
        ipa = ipa[:-1]
    # TODO? return utils.remove_non_lyric_punctuation(ipa)
    return ipa


def remove_stresses(text):
    return re.sub(r'\ˈ|\ˌ', '', text)


@lru_cache(maxsize=3000)
def to_ipa_tokens(text):
    '''Translate to IPA, returns list of words
       eng_to_ipa (lookup) -> custom lookup (via GPT) -> g2p (from letters)
    '''
    import eng_to_ipa

    # eng_to_ipa puts a * next to every word it can't translate
    text = utils.normalize_lyric(text)
    words: Any = eng_to_ipa.convert(text)

    ipa = []
    for w in words.split():
        if '*' in w or not w.strip():
            plain_word = w.replace('*', '').strip()
            w = utils.data.gpt_ipa.get(plain_word, None)
            if not w:
                w = get_g2p_ipa(plain_word)
                if not w:
                    print(f'WARNING! Could not get IPA for "{plain_word}" '
                          f'while translating "{text}"')
                    return ''
        ipa.append(fix_ipa_word(w))
    return ipa


def to_ipa(text):
    '''Translate to IPA, returns string
    '''
    return ' '.join(to_ipa_tokens(text))


def fix_ipa_word(w):
    '''Fixes some miscellanous IPA translation issues
    '''
    if not w:
        return ''
    w = re.sub(r"'ɛs$", "s", w)
    w = re.sub(r"'", "", w)  # remove apostraphe (not stress)
    return w.strip()


def get_g2p_ipa(text):
    '''Gets a IPA translation via g2p library (non-lookup)
    '''
    if text[-1] == "'":
        return re.sub(r'ŋ$', 'n', transducer()(text[:-1] + 'g').output_string)
    return transducer()(text).output_string


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
        elif char == "ˈ":
            features = [2.0] + utils.EMPTY_IPA_FEATURES
        elif char == "ˌ":
            features = [1.0] + utils.EMPTY_IPA_FEATURES
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


@lru_cache(maxsize=1000)
def get_ipa_vowel_vector(text, max_len=100):
    '''DB comparison vector for all vowels in a phrase
    '''
    ipa = to_ipa(text)
    vec = []
    for c in ipa:
        if is_vowel(c):
            ft = ftable().word_array([
                'son', 'cons', 'voi', 'long',
                'round', 'back', 'lo', 'hi', 'tense'
            ], c).tolist() or ([0.0] * 9)
            vec += ft
    vec = [item for sublist in vec for item in sublist][-max_len:]
    return vec


def get_stresses_vector(q):
    import pronouncing as pron
    stresses = []
    for word in q.split():
        word = re.sub(r'in\'', 'ing', word)
        p = pron.phones_for_word(word)
        p = p[0] if len(p) else ''
        if p:
            s = pron.stresses(p)
            stresses.append(s if len(s) else '0')
        else:
            stresses.append('1')
    return [int(s) for s in (''.join(stresses))]


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
