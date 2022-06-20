import re
import string
import spacy
import inflect
import sh
import eng_to_ipa
import pronouncing as pron
from functools import cache, lru_cache
from nltk import FreqDist
from nltk.util import ngrams as nltk_make_ngrams
from nltk.corpus import brown
from num2words import num2words

inflector = inflect.engine()


@cache
def get_common_words(n=700):
    fd = FreqDist(i.lower() for i in brown.words())
    return dict(fd.most_common()[:n])


@cache
def get_synonyms():
    with open('./data/synonyms.txt', 'r') as syn_file:
        return [
            [l.strip() for l in line.split(';')]
            for line in syn_file.readlines()
        ]


@lru_cache(maxsize=500)
def get_word_splits(word):
    splits = set()
    common = get_common_words()
    for sp in range(1, len(word)):
        w1 = word[:sp]
        w2 = word[sp:]
        if w1 in common and w2 in common:
            splits.add(' '.join([w1, w2]))
    return list(splits)


def tokenize_lyrics(lyrics, stop_words=None, unique=False):
    toks = [
        tok for tok in tokenize_lyric_line(' '.join(lyrics.split('\n')))
        if tok not in (stop_words or [])
    ]
    if unique:
        toks = list(set(toks))
    return toks


def tokenize_lyric_line(val):
    val = val.lower()
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


def get_lyric_ngrams(lyrics, n_range=range(5)):
    ngrams = []
    lines = [tokenize_lyric_line(l.strip()) for l in lyrics.split('\n') if l.strip()]
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
def make_synonyms(gram):
    all_syns = set()

    words = gram.split()
    for idx, word in enumerate(words):
        syns = set()

        if len(word) >= 5:
            if word.endswith('in\''):
                syns.add(re.sub(r"'$", "g", word))
                syns.add(word[:-1])
            elif word.endswith('ing'):
                syns.add(re.sub(r'g$', '\'', word))
                syns.add(word[:-1])
            elif word.endswith('in'):
                syns.add(word + 'g')
                syns.add(word + "'")

        if word.startswith('a-'):
            syns.add(word[2:])
            syns.add('a' + word[2:])

        if word.endswith("'ve") and len(word) > 3:
            if word[-4] == 'd':
                syns.add(word[:-3] + 'a')
            syns.add(word[:-3] + ' have')

        try:
            syns.add(num2words(word))
        except:
            pass

        # add plural/singular if it only involves adding/removing an s
        simple_plural = word + 's'
        simple_sing = word[:-1]

        if simple_plural == inflector.plural(word):
            syns.add(simple_plural)
        elif simple_sing == inflector.singular_noun(word):
            syns.add(simple_sing)

        match_w = word.lower().strip()
        matches = [line for line in get_synonyms() if match_w in line]
        for line in matches:
            for l in line:
                tok = l.lower().strip()
                if tok != match_w:
                    syns.add(tok)

        for splitw in get_word_splits(word):
            syns.add(splitw)

        for syn in syns:
            all_syns.add(re.sub(r'\b%s\b' % word, syn, str(gram)))

    if ' ' in gram:
        compound = re.sub(' ', '', gram)
        if compound in get_common_words():
            all_syns.add(compound)

    return [syn for syn in all_syns if syn != gram]


@lru_cache(maxsize=500)
def phones_for_word(w):
    w = re.sub(r'in\'', 'ing', w)
    val = pron.phones_for_word(w)
    return val[0] if len(val) else ''


@cache
def get_nlp():
    vocab = 'en_core_web_sm'
    try:
        nlp = spacy.load(vocab)
    except:
        spacy.cli.download(vocab)
        nlp = spacy.load(vocab)
    nlp.tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab)
    return nlp


@cache
def pstresses(w):
    val = pron.stresses(w)
    return val[0] if len(val) else ''


@lru_cache(maxsize=500)
def get_stresses(q):
    return ' '.join([pstresses(w) for w in q.split()])


@lru_cache(maxsize=500)
def get_phones(q, vowels_only=False, include_stresses=False, try_syns=True):
    if try_syns == True:
        try_syns = make_synonyms(q)
    phones = None

    for tok in [q] + list(try_syns or []):
        words = tok.split()
        word_phones = [phones_for_word(w) for w in words]
        word_phones = [p for p in word_phones if p]
        if len(word_phones) != len(words):
            continue

        phones = ' '.join(word_phones)
        if vowels_only:
            phones = re.sub(r'[A-Z]+\b', '', phones)
        if not include_stresses:
            phones = re.sub(r'[\d]+', '', phones)

        phones = re.sub(r'\s+', ' ', phones)
        phones = phones.strip()
        if phones:
            break

    return phones or ''


@lru_cache(maxsize=500)
def get_ipa(txt, phones=False, include_stresses=True):
    txt = txt.strip()
    txt = re.sub(r'-', ' ', txt)
    txt = re.sub(r'in\'', 'ing', txt)

    ipa = eng_to_ipa.convert(txt)

    # line = eng_to_ipa.ipa_list(txt)
    # line = ' '.join([ '|'.join(toks) for toks in line ])

    if '*' in ipa:
        return ''

    if phones:
        cmd = sh.python('-m',  'gruut_ipa', 'phones', ipa)
        cmd.wait()
        ipa_out = str(cmd)[:-1]
        if not include_stresses:
            ipa_out = re.sub(r'[\ˈˌ]', '', ipa_out)
        return ipa_out

    return ipa


@lru_cache(maxsize=500)
def get_formants(q, vowels_only=False, include_stresses=False):
    phones = get_ipa(q, phones=True).split()
    formants = []

    for phone in phones:
        if not include_stresses:
            phone = re.sub(r'[\ˈˌ]', '', phone)
        if phone in VOWELS or not vowels_only:
            f = PHONE_TO_FORMANTS.get(phone, [0, 0, 0, 0])
            formants.append(f)

    if len([f for f in formants if f != [0, 0, 0, 0]]) < (len(formants) / 2):
        return None

    return formants


@lru_cache(maxsize=500)
def get_mscore(text):
    mscore = [POS_TO_MSCORE.get(tok.pos_, 0) for tok in get_nlp()(text)]
    mscore[-1] *= 1.5
    return sum(mscore) / len(mscore)


POS_TO_MSCORE = dict(ADJ=4, NOUN=4, PROPN=4, VERB=4, ADV=2, ADP=2, INTJ=2, NUM=2, X=2, PRON=1)

VOWELS = [u'i', u'y', u'e', u'ø', u'ɛ', u'œ', u'a', u'ɶ', u'ɑ', u'ɒ', u'ɔ',
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


@cache
def get_idioms():
    with open('./data/idioms.txt', 'r') as f:
        return f.read()


@cache
def get_mine():
    with open('./data/mine.txt', 'r') as f:
        return f.read()
