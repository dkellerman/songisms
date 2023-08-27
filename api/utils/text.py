'''Text processing utilities'''

import re
import string
import inflect
import nltk
from functools import lru_cache
from minineedle import needle, core
import pronouncing as pron
from nltk.util import ngrams as nltk_make_ngrams
from num2words import num2words
from api.utils.data import get_common_words, get_custom_variants, get_sim_sounds

inflector = inflect.engine()


def tokenize_lyric(val):
    '''Split lyric into normalized tokens'''
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
    '''Normalize lyric without tokenizing'''
    return ' '.join(tokenize_lyric(val))


def remove_stresses(text):
    '''Remove IPA stress marks'''
    return re.sub(r'\ˈ|\ˌ', '', text)


def remove_punctuation(text):
    '''Remove all punctuation'''
    return ''.join([t for t in text if t not in string.punctuation])


def remove_non_lyric_punctuation(text):
    '''Remove punctuation not critical to lyrics'''
    return ''.join([t for t in text if t not in r"""!"#$%&()*+,./:;<=>?@[\]^_`{|}~"""])


def align_vals(val1, val2):
    '''Aligns two strings or lists of strings using Needleman-Wunsch.
       Returned sequences may contain gap character classes - calling str()
       will convert them to a gap character (_)
    '''
    aligner = needle.NeedlemanWunsch(val1, val2)
    aligner.gap_character = '_'
    aligner.align()
    fmt = core.AlignmentFormat.str if type(val1) == str else core.AlignmentFormat.list
    aligned_val1, aligned_val2 = aligner.get_aligned_sequences(fmt)
    return aligned_val1, aligned_val2, aligner


@lru_cache(maxsize=500)
def make_variants(gram):
    '''Make a list of variants of a word or phrase for searching'''
    words = gram.split()

    if ' ' in gram:
        compound = re.sub(' ', '', gram)
        if compound in get_common_words():
            words.append(compound)

    variants = []
    for word in words[-1:]:
        if len(word) >= 5:
            if word.endswith('in\''):
                variants.append(re.sub(r"'$", "g", word))
                variants.append(word[:-1])
            elif word.endswith('ing'):
                variants.append(re.sub(r'g$', '\'', word))
                variants.append(word[:-1])
            elif word.endswith('in'):
                variants.append(word + 'g')
                variants.append(word + "'")

        if word.startswith('a-'):
            variants.append(word[2:])
            variants.append('a' + word[2:])

        if word.endswith("'ve") and len(word) > 3:
            if word[-4] == 'd':
                variants.append(word[:-3] + 'a')
            variants.append(word[:-3] + ' have')

        try:
            variants.append(num2words(word))
        except:
            pass

        # add plural/singular if it only involves adding/removing an s
        simple_plural = word + 's'
        simple_sing = word[:-1]

        if simple_plural == inflector.plural(word):
            variants.append(simple_plural)
        elif simple_sing == inflector.singular_noun(word):
            variants.append(simple_sing)

        match_w = word.lower().strip()
        matches = [line for line in get_custom_variants() if match_w in line]
        for line in matches:
            for l in line:
                tok = l.lower().strip()
                if tok != match_w:
                    variants.append(tok)

    variants = list(set(variants))

    for var in [gram] + variants:
        rhymes = pron.rhymes(var)
        variants += [r for r in rhymes if r in get_common_words()]
        variants += get_word_splits(var)
        variants += [ss.lower() for ss in get_sim_sounds().get(var, [])]

    variants = set(variants)
    return [var for var in variants if var != gram]


@lru_cache(maxsize=500)
def get_word_splits(word):
    '''Return possible splits of a word into two words'''
    splits = set()
    common = get_common_words()
    for sp in range(1, len(word)):
        w1 = word[:sp]
        w2 = word[sp:]
        if w1 in common and w2 in common:
            splits.add(' '.join([w1, w2]))
    return list(splits)


def get_lyric_ngrams(lyrics, n_range=range(5)):
    '''Return all possible word-based ngrams of all lengths in a given range'''
    ngrams = []
    lines = [tokenize_lyric(l.strip())
             for l in lyrics.split('\n') if l.strip()]
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
            tuple(sorted((a.strip().lower(), b.strip().lower(),)))
            for idx, a in enumerate(grams) for b in grams[idx + 1:]
        ]
    return list(set(pairs))


def make_homophones(w, ignore_stress=True, multi=True):
    '''Get homophones'''
    # currently needs to be installed via `pip install homophones``
    from homophones import homophones as hom

    w = re.sub(r'in\'', 'ing', w)
    all_words = list(hom.Words_from_cmudict_string(hom.ENTIRE_CMUDICT))
    words = list(hom.Word.from_string(w, all_words))
    results = []
    common = get_common_words()
    for word in words:
        for h in hom.homophones(word, all_words, ignore_stress=ignore_stress):
            if h.word.lower() != w and h.word.lower() in common:
                results.append(h.word.lower())
        if multi:
            for h in hom.dihomophones(word, all_words, ignore_stress=ignore_stress):
                if all([x.word.lower() in common for x in h]):
                    results.append(' '.join([x.word.lower() for x in h]))
            for h in hom.trihomophones(word, all_words, ignore_stress=ignore_stress):
                if all([x.word.lower() in common for x in h]):
                    results.append(' '.join([x.word.lower() for x in h]))

    return [r for r in results if '(' not in r]


@lru_cache(maxsize=500)
def get_mscore(text):
    '''Meaning score is a custom metric for how meaningful a lyric is based on part-of-speech.
    '''
    toks = tokenize_lyric(text)
    pos = nltk.pos_tag(toks)
    mscore = [POS_TO_MSCORE.get(tok[1], 0) for tok in pos if tok[1]]
    mscore[-1] *= 1.5
    return sum(mscore) / len(mscore)


POS_TO_MSCORE = dict(CC=2, CD=1, DT=1, EX=1, IN=2, JJ=4, JJR=4, JJS=4, LS=1, MD=2, NN=4, NNP=3, NNPS=3,
                     NNS=3, PDT=2, POS=0, PRP=2, RB=4, RBR=4, RBS=4, RP=3, TO=1, UH=3, VB=4,
                     VBD=4, VBG=4, VBN=4, VBP=4, VBZ=4, WDT=2, WP=2, WRB=2, SYM=0)
POS_TO_MSCORE['PRP$'] = 2
POS_TO_MSCORE['WP$'] = 2
POS_TO_MSCORE["''"] = 2
