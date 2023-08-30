'''IPA and other pronounciation-related utilities
'''

import re
from functools import lru_cache, cached_property
from collections import defaultdict
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
    ipa = ipa.strip().replace("ː", "")
    return utils.remove_non_lyric_punctuation(ipa)


def remove_stresses(text):
    '''Remove IPA stress marks
    '''
    return re.sub(r'\ˈ|\ˌ', '', text)


def get_ipa_words(text):
    '''Translate to IPA, returns list of words
       eng_to_ipa (lookup) -> custom lookup (via GPT) -> g2p (from sounds)
    '''
    import eng_to_ipa
    words = eng_to_ipa.convert(text).split()
    ipa = []
    for w in words:
        if '*' in w or not w.strip():
            w = w.replace('*', '').strip()
            w = utils.data.gpt_ipa.get(w, get_g2p_word(w))
        ipa.append(fix_ipa_word(w))
    return ipa


def get_ipa_text(text):
    '''Translate to IPA, returns string
    '''
    return ' '.join(get_ipa_words(text))


def fix_ipa_word(w):
    '''Fixes some miscellanous IPA translation issues
    '''
    if w is None:
        return ''
    w = re.sub(r"'ɛs$", "s", w)
    w = re.sub(r"'", "", w)
    return w.strip()


def get_g2p_word(w):
    '''Gets a IPA translation via g2p library (based on sounds, not lookup)
    '''
    if w[-1] == "'":
        return re.sub(r'ŋ$', 'n', transducer()(w[:-1] + 'g').output_string)
    return transducer()(w).output_string


def get_ipa_features(ipa_letter):
    '''Get feature table for IPA character
    '''
    f = ftable().fts(ipa_letter)
    return f


def is_vowel(ipa_letter):
    return ipa_letter in IPA_VOWELS


def get_stress_tail_for_ipa(ipa_phrase):
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


def get_ipa_tail(text, stresses=False):
    '''Convert word to IPA and return the stress tail only
    '''
    text = text.lower().strip()
    ipa = get_ipa_text(text)
    tail = get_stress_tail_for_ipa(ipa)
    val = ''.join(tail.split(' '))
    if not stresses:
        val = remove_stresses(val)
    return val


@lru_cache(maxsize=1000)
def get_vowel_vector(text, max_len=100):
    '''DB comparison vector for vowels in a word
    '''
    ipa = get_ipa_tail(text)
    vec = []
    for c in ipa:
        if is_vowel(c):
            ft = ftable().word_array([
                'syl', 'son', 'cons', 'voi', 'long',
                'round', 'back', 'lo', 'hi', 'tense'
            ], c).tolist() or ([0.0] * 10)
            vec += ft
    vec = [item for sublist in vec for item in sublist][-max_len:]
    return vec


FULL_IPA_FEATURE_LEN = len(get_ipa_features('a').numeric())
EMPTY_FEATURES = [0.0] * FULL_IPA_FEATURE_LEN

def get_rhyme_vectors(text1, text2):
    '''Aligns IPA stress tails and returns full feature vectors for each
    '''
    ipa1 = get_ipa_tail(text1)
    ipa2 = get_ipa_tail(text2)
    seq1, seq2, _ = utils.align_vals(ipa1, ipa2)
    vec1, vec2 = [], []

    for idx, c in enumerate(seq1):
        if c == '-' or seq2[idx] == '-' or c == '' or seq2[idx] == '':
            vec1 += EMPTY_FEATURES
            vec2 += EMPTY_FEATURES
        else:
            ft1 = get_ipa_features(c)
            ft2 = get_ipa_features(seq2[idx])
            if ft1 is None or ft2 is None:
                vec1 += EMPTY_FEATURES
                vec2 += EMPTY_FEATURES
            else:
                vec1 += [float(f) for f in ft1.numeric()]
                vec2 += [float(f) for f in ft2.numeric()]
    return vec1, vec2


def score_rhyme(text1, text2):
    from scipy.spatial import distance
    vec1, vec2 = get_rhyme_vectors(text1, text2)
    score = distance.cosine(vec1, vec2)
    return score


ipa_token_dict = defaultdict(lambda: len(ipa_token_dict))

def ipa_to_unique_tokens(text):
    return [ipa_token_dict[c] for c in text]


@lru_cache(maxsize=1000)
def get_stresses_vector(q):
    '''Get stresses vector
    '''
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


@lru_cache(maxsize=1000)
def get_formants_vector(q, vowels_only=False, include_stresses=False):
    ipa = get_ipa_text(q).split()
    formants = []

    for ch in ipa:
        if not include_stresses:
            ch = re.sub(r'[\ˈˌ]', '', ch)
        if ch in IPA_VOWELS or not vowels_only:
            f = PHONE_TO_FORMANTS.get(ch, [0, 0, 0, 0])
            formants.append(f)

    if len([f for f in formants if f != [0, 0, 0, 0]]) < (len(formants) / 2):
        return None

    return formants



IPA_VOWELS = [u'i', u'y', u'e', u'ø', u'ɛ', u'œ', u'a', u'ɶ', u'ɑ', u'ɒ', u'ɔ',
              u'ʌ', u'ɤ', u'o', u'ɯ', u'u', u'ɪ', u'ʊ', u'ə', u'æ']

PHONE_TO_FORMANTS = {
    u'i': [240, 2400, 2160, 0],
    u'y': [235, 2100, 1865, 0],
    u'e': [390, 2300, 1910, 0],
    u'ø': [370, 1900, 1530, 0],
    u'ɛ': [610, 1900, 1290, 0],
    u'œ': [585, 1710, 1125, 0],
    u'a': [850, 1610, 760, 0],
    u'ɶ': [820, 1530, 710, 0],
    u'ɑ': [750, 940, 190, 0],
    u'ɒ': [700, 760, 60, 0],
    u'ɔ': [500, 700, 200, 0],
    u'ʌ': [600, 1170, 570, 0],
    u'ɤ': [460, 1310, 850, 0],
    u'o': [360, 640, 280, 0],
    u'ɯ': [300, 1390, 1090, 0],
    u'u': [250, 595, 345, 0],

    # https://www.jobilize.com/course/section/formant-analysis-lab-9a-speech-processing-part-1-by-openstax#uid32
    u'ɪ': [390, 1990, 2550, 0],
    u'ʊ': [300, 870, 2240, 0],
    u'ə': [520, 1190, 2390, 0],
    u'æ': [660, 1720, 2410, 0],

    # https://www.advancedbionics.com/content/dam/advancedbionics/Documents/libraries/Tools-for-Schools/Educational_Support/Tools-for-Learning-about-Hearing-loss-and-Cochlear-Implants/ToolsforSchools-Sounds-of-Speech-Flyer.pdf
    u'ŋ': [325, 1250, 2500, 0],
    u'ʧ': [0, 0, 1750, 4500],
    u'θ': [0, 0, 0, 6000],
    u'ð': [300, 0, 0, 5250],
    u'ʤ': [250, 0, 2500, 0],
    u'ʃ': [0, 0, 1750, 5000],
    u'w': [525, 0, 0, 0],
    u'n': [300, 1250, 2500, 0],
    u'm': [300, 1250, 3000, 0],
    u'r': [700, 2250, 2100, 0],
    u'g': [250, 0, 2000, 0],
    u'j': [250, 0, 2500, 0],
    u'l': [325, 0, 2500, 0],
    u'd': [350, 0, 2750, 0],  # ???
    u'z': [250, 0, 0, 4500],
    u'v': [350, 0, 0, 4000],
    u'h': [0, 0, 1750, 0],
    u'p': [0, 0, 1750, 0],
    u'k': [0, 0, 2250, 0],
    u't': [0, 0, 3000, 0],
    u's': [0, 0, 0, 5500],
    u'f': [0, 0, 0, 4500],
    u'b': [350, 0, 2250, 0],
    # ??? u'a': [],
}
