'''Text processing utilities related to lyrics
'''

import re
import string
from functools import lru_cache
from typing import List
from songisms import utils


@lru_cache(maxsize=None)
def inflector():
    import inflect
    return inflect.engine()


def tokenize_lyric(val):
    '''Split lyric into normalized tokens
    '''
    val = val.lower().strip()
    val = re.sub(r'\s+', ' ', val)  # multiple spaces to one
    val = val.replace('&', 'and')
    val = remove_non_lyric_punctuation(val)
    toks = [t for t in val.split()]
    toks = [strip_periods_preserve_acronyms(t) for t in toks]
    return toks


def normalize_lyric(val):
    '''Normalize lyric without returning tokens array
    '''
    return ' '.join(tokenize_lyric(val))


def strip_periods_preserve_acronyms(s):
    if len(s) <= 2:
        return s.replace('.', '')
    if re.match(r'([A-Za-z](\.|$)+)+$', s) and not re.match(r'[A-Za-z]\.$', s):
        return '.'.join([ch for ch in s if ch != '.']) + '.'
    return s.replace('.', '')


def test_normalize_lyric():
    assert(strip_periods_preserve_acronyms("L.A.") == "L.A.")
    assert(strip_periods_preserve_acronyms("L.A") == "L.A.")
    assert(strip_periods_preserve_acronyms("LA.") == "LA")
    assert(strip_periods_preserve_acronyms("LA...") == "LA")
    assert(strip_periods_preserve_acronyms("L.A...") == "L.A.")
    assert(strip_periods_preserve_acronyms("la...") == "la")
    assert(strip_periods_preserve_acronyms("l.a...") == "l.a.")
    assert(strip_periods_preserve_acronyms("l.") == "l")
    assert(strip_periods_preserve_acronyms("i") == "i")
    assert(strip_periods_preserve_acronyms(".") == "")
    assert(strip_periods_preserve_acronyms("...") == "")
    assert(normalize_lyric("la-la-la-la") == "la-la-la-la")
    assert(normalize_lyric("v.") == "v")
    assert(normalize_lyric("C-O-L-A Cola") == "c-o-l-a cola")
    assert(normalize_lyric("Another Day in L.A.!") == "another day in l.a.")
    assert(normalize_lyric("Another Day in L.A....") == "another day in l.a.")
    assert(normalize_lyric("  Another Day in L.A.... hi just chillin' ") ==
                           "another day in l.a. hi just chillin'")
    assert(normalize_lyric("Another Day in LA...") == "another day in la")
    assert(normalize_lyric("'cause we won't be gettin' a-goin'! (ok?) [**]") ==
                           "'cause we won't be gettin' a-goin' ok")
    assert(normalize_lyric("Here we go!!   Lotsa punctuation & spacing... ok?!?!") ==
                           "here we go lotsa punctuation and spacing ok")


def remove_all_punctuation(text):
    '''Remove all punctuation
    '''
    return ''.join([t for t in text if t not in string.punctuation])


def remove_non_lyric_punctuation(text):
    '''Remove punctuation not critical to lyrics
    '''
    return ''.join([t for t in text if t not in r"""!"#$%&()*+,/:;<=>?@[\]^_`{|}~"""])


def align_vals(val1, val2):
    '''Aligns two strings or lists of strings using Needleman-Wunsch.
       Returned sequences may contain gap character classes - calling str()
       will convert them to a gap character (_)
    '''
    from minineedle import needle, core
    aligner = needle.NeedlemanWunsch(val1, val2)
    aligner.gap_character = '_'
    aligner.align()
    fmt = core.AlignmentFormat.str if type(val1) == str else core.AlignmentFormat.list
    aligned_val1, aligned_val2 = aligner.get_aligned_sequences(fmt)
    return aligned_val1, aligned_val2, aligner


@lru_cache(maxsize=1000)
def get_variants(gram, spelling=True):
    '''Make a list of variants of a word or phrase for searching
    '''
    import pronouncing as pron
    from num2words import num2words

    words = gram.split()
    if ' ' in gram:
        compound = re.sub(' ', '', gram)
        if compound in utils.data.common_words:
            words.append(compound)

    variants = []
    word = words[-1] # just using the last word works best for now

    # common misspellings
    if spelling:
        new_sp = utils.data.misspellings.get(word, None)
        if new_sp:
            variants.append(new_sp)
            word = new_sp  # make variants of this

    # in' -> ing
    if len(word) >= 5:
        if word.endswith('in\''):
            variants.append(re.sub(r"'$", "g", word))
            variants.append(word[:-1])
            variants.append(word[:-3] + ' in')
            variants.append(word[:-3] + 'en')
        elif word.endswith('ing'):
            variants.append(re.sub(r'g$', '\'', word))
            variants.append(word[:-1])
        elif word.endswith('in'):
            variants.append(word + 'g')
            variants.append(word + "'")
        elif word.endswith('en'):
            variants.append(word[:-2] + 'ing')
            variants.append(word[:-2] + "in'")

    # froggie went a-courtin'
    if word.startswith('a-') and len(word) > 5:
        variants.append(word[2:])
        variants.append('a' + word[2:])

    #  would've woulda would have
    if word.endswith("'ve") and len(word) > 3:
        if word[-4] == 'd':
            variants.append(word[:-3] + 'a')
        variants.append(word[:-3] + ' have')

    # try numbers as words also
    try:
        variants.append(num2words(word))
    except:
        pass

    # add plural/singular if it only involves adding/removing an s
    simple_plural = word + 's'
    simple_sing = word[:-1]
    if simple_plural == inflector().plural(word):
        variants.append(simple_plural)
    elif simple_sing == inflector().singular_noun(word):
        variants.append(simple_sing)

    # more additive/subtractive variants
    if len(word) > 3:
        if word.endswith('e'):
            variants.append(word + 'd')
        elif word.endswith('ed'):
            variants.append(word[:-1])
        elif word.endswith('et'):
            variants.append(word[:-1] + 'd')

    # custom variants, mostly very songy kinda stuff
    match_w = word.lower().strip()
    matches = [line for line in utils.data.custom_variants if match_w in line]
    for line in matches:
        for l in line:
            tok = l.lower().strip()
            if tok != match_w:
                variants.append(tok)

    # reprocess all the words
    variants = list(set(variants))
    for var in [gram] + variants:
        # add some stupid rhymes for lookup sake, seems to help
        rhymes = pron.rhymes(var)
        variants += [r for r in rhymes if r in utils.data.common_words]
        # compound word splits
        variants += get_word_splits(var)
        # similar sounding words, covers heterophones
        variants += [ss.lower() for ss in utils.data.sim_sounds.get(var, [])]

    variants = set(variants)
    return [var for var in variants
            if var != gram
            # avoid homographs, e.g. "read"
            and len(pron.phones_for_word(var)) == 1]


@lru_cache(maxsize=1000)
def get_word_splits(word):
    '''Return possible splits of a word into two words
    '''
    splits = set()
    for sp in range(1, len(word)):
        w1 = word[:sp]
        w2 = word[sp:]
        if w1 in utils.data.common_words and w2 in utils.data.common_words:
            splits.add(' '.join([w1, w2]))
    return list(splits)


def get_lyric_ngrams(lyrics, n_range=range(5)):
    '''Return all possible word-based ngrams of all lengths in a given range.
       Lyrics should be passed in as one big newline separated string.
    '''
    from nltk.util import ngrams as nltk_make_ngrams

    ngrams = []
    lines = [tokenize_lyric(l.strip()) for l in lyrics.split('\n') if l.strip()]
    for toks in lines:
        for i in n_range:
            for ngram in nltk_make_ngrams(sequence=toks, n=i+1):
                ngrams.append((' '.join(ngram), i+1))
    return ngrams


def get_rhyme_pairs(val=''):
    '''Takes a list of rhyme sets separated by newlines, with each rhyme separated by semicolons.
       Outputs all possible pairs of rhymes.
    '''
    lines = val.strip().split('\n')
    pairs = []
    for line in lines:
        grams = line.split(';')
        pairs += [
            tuple(sorted((normalize_lyric(a), normalize_lyric(b))))
            for idx, a in enumerate(grams) for b in grams[idx + 1:]
        ]
    return list(set(pairs))


def get_mscore(text):
    '''Meaning score is a custom metric for how meaningful a lyric is based on part-of-speech.
    '''
    import nltk

    toks = tokenize_lyric(text)
    pos = nltk.pos_tag(toks)
    mscore: List[float] = [POS_TO_MSCORE.get(tok[1], 0) for tok in pos if tok[1]]
    if len(toks) > 1:
        mscore[-1] *= 1.3
    return sum(mscore) / len(mscore)


POS_TO_MSCORE = dict(CC=2, CD=1, DT=1, EX=1, IN=2, JJ=4, JJR=4, JJS=4, LS=1, MD=2, NN=4, NNP=3,
                     NNPS=3, NNS=3, PDT=2, POS=0, PRP=2, RB=4, RBR=4, RBS=4, RP=3, TO=1, UH=3,
                     VB=4, VBD=4, VBG=4, VBN=4, VBP=4, VBZ=4, WDT=2, WP=2, WRB=2, SYM=0)
POS_TO_MSCORE['PRP$'] = 2
POS_TO_MSCORE['WP$'] = 2
POS_TO_MSCORE["''"] = 2

test_normalize_lyric()
