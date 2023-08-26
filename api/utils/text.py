import re
import g2p
import eng_to_ipa
import string
import json
import inflect
import spacy
from scipy.spatial import distance
from functools import lru_cache
from panphon import featuretable
from minineedle import needle, core
import pronouncing as pron
from nltk import FreqDist
from nltk.util import ngrams as nltk_make_ngrams
from nltk.corpus import brown
from num2words import num2words


ftable = featuretable.FeatureTable()
transducer = g2p.make_g2p('eng', 'eng-ipa')
inflector = inflect.engine()


@lru_cache(maxsize=None)
def get_common_words(n=700):
    fd = FreqDist(i.lower() for i in brown.words())
    return dict(fd.most_common()[:n])



@lru_cache(maxsize=None)
def get_custom_variants():
    with open('./data/variants.txt', 'r') as syn_file:
        return [
            [l.strip() for l in line.split(';')]
            for line in syn_file.readlines()
        ]


@lru_cache(maxsize=None)
def get_sim_sounds():
    with open('./data/simsounds.json', 'r') as f:
        return json.loads(f.read())


@lru_cache(maxsize=None)
def get_gpt_ipa():
    with open('./data/ipa_gpt.json', 'r') as f:
        return json.load(f)


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


def get_ipa_text(text):
    return ' '.join(get_ipa_words(text))


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
    return ipa_letter in IPA_VOWELS


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
    aligner = needle.NeedlemanWunsch(val1, val2)
    aligner.gap_character = '_'
    aligner.align()
    fmt = core.AlignmentFormat.str if type(val1) == str else core.AlignmentFormat.list
    aligned_val1, aligned_val2 = aligner.get_aligned_sequences(fmt)
    score = aligner.get_score()
    return aligned_val1, aligned_val2, score, aligner


def get_ipa_tail(text, stresses=False):
    text = text.lower().strip()
    ipa = get_ipa_words(text)
    text = ' '.join(ipa)
    tail = get_stress_tail(text)
    val = ''.join(tail.split(' '))
    if not stresses:
        val = remove_stresses(val)
    return val


@lru_cache(maxsize=500)
def get_vowel_vector(text, max_len=100):
    global ftable
    ipa = get_ipa_tail(text)
    vec = []
    for c in ipa:
        if is_vowel(c):
            ft = ftable.word_array([
                'syl', 'son', 'cons', 'voi', 'long',
                'round', 'back', 'lo', 'hi', 'tense'
            ], c).tolist() or ([0] * 10)
            vec += ft
    vec = [item for sublist in vec for item in sublist][-max_len:]
    return vec


def score_rhyme(text1, text2):
    ipa1 = get_ipa_tail(text1)
    ipa2 = get_ipa_tail(text2)
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



@lru_cache(maxsize=500)
def make_variants(gram):
    words = gram.split()

    if ' ' in gram:
        compound = re.sub(' ', '', gram)
        if compound in get_common_words():
            words.append(compound)

    variants = []
    for word in words:
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

    # ??? for var in variants: variants.add(re.sub(r'\b%s\b' % word, var, str(gram)))


@lru_cache(maxsize=500)
def get_word_splits(word):
    splits = set()
    common = get_common_words(1000)
    for sp in range(1, len(word)):
        w1 = word[:sp]
        w2 = word[sp:]
        if w1 in common and w2 in common:
            splits.add(' '.join([w1, w2]))
    return list(splits)


def get_lyric_ngrams(lyrics, n_range=range(5)):
    ngrams = []
    lines = [tokenize_lyric(l.strip()) for l in lyrics.split('\n') if l.strip()]
    for toks in lines:
        for i in n_range:
            for ngram in nltk_make_ngrams(sequence=toks, n=i+1):
                ngrams.append((' '.join(ngram), i+1))
    return ngrams


def get_rhyme_pairs(val=''):
    lines = val.strip().split('\n')
    pairs = []
    for line in lines:
        grams = line.split(';')
        pairs += [
            tuple(sorted((a.strip().lower(), b.strip().lower(),)))
            for idx, a in enumerate(grams) for b in grams[idx + 1:]
        ]
    return list(set(pairs))


@lru_cache(maxsize=500)
def make_homophones(w, ignore_stress=True, multi=True):
    # currently needs to be installed via `pip install homophones``
    from homophones import homophones as hom

    w = re.sub(r'in\'', 'ing', w)
    all_words = list(hom.Words_from_cmudict_string(hom.ENTIRE_CMUDICT))
    words = list(hom.Word.from_string(w, all_words))
    results = []
    common = brown.words()
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


@lru_cache(maxsize=None)
def get_nlp():
    vocab = 'en_core_web_sm'
    try:
        nlp = spacy.load(vocab)
    except:
        spacy.cli.download(vocab)
        nlp = spacy.load(vocab)
    nlp.tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab)
    return nlp


@lru_cache(maxsize=500)
def get_stresses(q):
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


@lru_cache(maxsize=500)
def get_formants(q, vowels_only=False, include_stresses=False):
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


@lru_cache(maxsize=500)
def get_mscore(text):
    mscore = [POS_TO_MSCORE.get(tok.pos_, 0) for tok in get_nlp()(text)]
    mscore[-1] *= 1.5
    return sum(mscore) / len(mscore)


POS_TO_MSCORE = dict(ADJ=4, NOUN=4, VERB=4, PROPN=3, ADV=2, ADP=2, INTJ=2, NUM=2, X=2, PRON=1)

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


@lru_cache(maxsize=None)
def get_idioms():
    with open('./data/idioms.txt', 'r') as f:
        return f.read()


@lru_cache(maxsize=None)
def get_mine():
    with open('./data/mine.txt', 'r') as f:
        return f.read()


IPA_VOWELS = [u'i', u'y', u'e', u'ø', u'ɛ', u'œ', u'a', u'ɶ', u'ɑ', u'ɒ', u'ɔ',
              u'ʌ', u'ɤ', u'o', u'ɯ', u'u', u'ɪ', u'ʊ', u'ə', u'æ']
