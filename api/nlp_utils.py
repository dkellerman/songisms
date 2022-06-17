import re
import string
import spacy
import inflect
import sh
import eng_to_ipa
import pronouncing as pron
from tqdm import tqdm
from nltk import FreqDist
from nltk.util import ngrams as nltk_make_ngrams
from nltk.corpus import brown
from num2words import num2words
from django.db import transaction
from django.db.models import Sum
import api.models as models

_syn_data = None
_common_words = None
inflector = inflect.engine()


def get_common_words():
    global _common_words
    if _common_words is not None:
        return _common_words
    flist = FreqDist(i.lower() for i in brown.words())
    _common_words = [w[0] for w in flist.most_common()[:1000]]
    return _common_words


def get_synonyms():
    global _syn_data
    if _syn_data is not None:
        return _syn_data
    with open('./data/synonyms.txt', 'r') as syn_file:
        _syn_data = [
            [l.strip() for l in line.split(';')]
            for line in syn_file.readlines()
        ]
    return _syn_data


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
    text = ' '.join([l for l in lyrics.split('\n') if l.strip()])
    toks = tokenize_lyric_line(text)
    for i in n_range:
        for ngram in nltk_make_ngrams(sequence=toks, n=i+1):
            ngrams.append((' '.join(ngram), i+1))
    return ngrams


def make_extra_ngrams(force_update=False):
    with open('./data/mine.txt', 'r') as f:
        lines = f.read().split('\n')
        make_ngrams_for_lyrics(lines, None, force_update=force_update)

    with open('./data/idioms.txt', 'r') as f:
        lines = [l.strip() for l in f.read().split('\n') if l.strip()]
        make_ngrams_for_lyrics(lines, None, force_update=force_update)


def make_ngrams_for_song(song, force_update=False):
    with transaction.atomic():
        models.SongNGram.objects.filter(song=song).delete()
        make_ngrams_for_lyrics(song.lyrics, song, force_update=force_update)


POS_TO_MSCORE = dict(ADJ=4, NOUN=4, PROPN=4, VERB=4, ADV=2, ADP=2, INTJ=2, NUM=2, X=2, PRON=1)


def make_ngrams_for_lyrics(lyrics, song=None, force_update=False):
    if type(lyrics) == str:
        lyrics = [lyrics]

    print('creating ngrams...')
    nlp = load_nlp()
    with transaction.atomic():
        sn_to_update = set()
        n_to_update = set()
        for lyric in lyrics:
            ngrams = get_lyric_ngrams(lyric, range(5))
            for key, n in ngrams:
                ngram, _ = models.NGram.objects.get_or_create(text=key, n=n)
                if not ngram.phones or force_update:
                    ngram.phones = get_phones(ngram.text, vowels_only=True, include_stresses=False)
                    n_to_update.add(ngram)
                # if not ngram.ipa or force_update:
                #     ngram.ipa = get_ipa(ngram.text)
                #     n_to_update.add(ngram)
                if not ngram.mscore or force_update:
                    mscore = [POS_TO_MSCORE.get(tok.pos_, 0) for tok in nlp(key)]
                    mscore[-1] *= 1.5
                    ngram.mscore = sum(mscore) / len(mscore)
                    n_to_update.add(ngram)
                if song:
                    sn, created = models.SongNGram.objects.get_or_create(ngram=ngram, song=song, defaults=dict(count=1))
                    if not created:
                        sn.count += 1
                        sn_to_update.add(sn)
        print('saving', len(n_to_update), 'ngrams...')
        models.SongNGram.objects.bulk_update(list(sn_to_update), ['count'])
        models.NGram.objects.bulk_update(list(n_to_update), ['phones', 'ipa', 'mscore'])


def score_ngrams(force_update=False):
    ngrams = models.NGram.objects.annotate(
        ct=Sum('song_ngrams__count'),
    )

    words = list(ngrams.filter(n=1))
    word_ct = sum([n.ct or 0 for n in words])
    by_text = dict([(n.text, n) for n in ngrams])
    n_counts = [sum([gram.ct or 0 for gram in ngrams if gram.n == n]) for n in range(15)]
    ngrams = list(ngrams)
    to_update = [n for n in ngrams if n.pct is None and n.adj_pct is None] if not force_update else ngrams

    print("[SCORING NGRAMS]", len(to_update))
    for ngram in tqdm(to_update):
        if not ngram.pct or not ngram.adj_pct or force_update:
            if ngram.n > 1:
                subgrams = [by_text[gram] for gram in ngram.text.split() if gram and (gram in by_text)]
                total_with_same_n = n_counts[ngram.n - 1]
                ngram_pct = float((ngram.ct or 0.0) / total_with_same_n)
                chance_pct = 1.0
                for gram in subgrams:
                    gram_pct = float((gram.ct or 0.0) / word_ct)
                    chance_pct *= gram_pct
                ngram.pct = ngram_pct
                ngram.adj_pct = ngram_pct - chance_pct
            else:
                ngram.pct = float((ngram.ct or 0.0) / word_ct)
                ngram.adj_pct = float((ngram.ct or 0.0) / word_ct)

    print("updating...")
    models.NGram.objects.bulk_update(to_update, ['pct', 'adj_pct'])


def set_lyrics_ipa(song):
    lines = song.lyrics.split('\n')
    ipa = '\n'.join([get_ipa(l, phones=False) for l in lines])
    song.lyrics_ipa = ipa
    song.save()


def make_rhymes(song):
    rhymes = get_rhyme_pairs(song.rhymes_raw)

    for text1, text2 in rhymes:
        n1 = len(text1.split())
        n2 = len(text2.split())
        with transaction.atomic():
            ngram1, _ = models.NGram.objects.get_or_create(text=text1, n=n1)
            ngram2, _ = models.NGram.objects.get_or_create(text=text2, n=n2)
            models.Rhyme.objects.get_or_create(
                from_ngram=ngram1,
                to_ngram=ngram2,
                song=song,
                level=1,
            )
            models.Rhyme.objects.get_or_create(
                from_ngram=ngram2,
                to_ngram=ngram1,
                song=song,
                level=1,
            )


def make_rhymes_l2():
    l1 = models.Rhyme.objects.distinct('from_ngram', 'to_ngram').filter(level=1)
    l2 = models.Rhyme.objects.distinct('from_ngram', 'to_ngram').filter(level=2)
    created = [(r.from_ngram_id, r.to_ngram_id) for r in l2]

    for r1 in tqdm(l1):
        for r2 in l1.filter(from_ngram_id=r1.to_ngram_id):
            with transaction.atomic():
                id1 = r1.from_ngram_id
                id2 = r2.to_ngram_id
                if (id1, id2) not in created:
                    models.Rhyme.objects.create(
                        from_ngram_id=id1,
                        to_ngram_id=id2,
                        song=None,
                        level=2,
                    )
                    created.append((id1, id2))

                if (id2, id1) not in created:
                    models.Rhyme.objects.create(
                        from_ngram_id=id2,
                        to_ngram_id=id1,
                        song=None,
                        level=2,
                    )
                    created.append((id2, id1))


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


def phones_for_word(w):
    w = re.sub(r'in\'', 'ing', w)
    val = pron.phones_for_word(w)
    return val[0] if len(val) else ''


def load_nlp():
    vocab = 'en_core_web_sm'
    try:
        nlp = spacy.load(vocab)
    except:
        spacy.cli.download(vocab)
        nlp = spacy.load(vocab)

    nlp.tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab)
    return nlp


def pstresses(w):
    val = pron.stresses(w)
    return val[0] if len(val) else ''


def get_stresses(q):
    return ' '.join([pstresses(w) for w in q.split()])


def get_phones(q, vowels_only=False, include_stresses=False, try_syns=True):
    if try_syns == True:
        try_syns = make_synonyms(q)
    phones = None

    for tok in [q] + (try_syns or []):
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


GRUUT_CACHE = {}


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
        if ipa in GRUUT_CACHE:
            return GRUUT_CACHE[ipa]
        cmd = sh.python('-m',  'gruut_ipa', 'phones', ipa)
        cmd.wait()
        ipa_out = str(cmd)[:-1]
        if not include_stresses:
            ipa_out = re.sub(r'[\ˈˌ]', '', ipa_out)
        GRUUT_CACHE[ipa] = ipa_out
        return ipa_out

    return ipa


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
