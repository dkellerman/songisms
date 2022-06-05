import re
import os
import time
import pafy
import ffmpeg
import json
import base64
import spacy
import graphene
import inflect
import sh
import eng_to_ipa
import pronouncing as pron
import firebase_admin
from nltk.util import ngrams as nltk_make_ngrams
from firebase_admin import firestore, credentials
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.core.files import File
from django.db import transaction
from django.db.models import Count
from django.conf import settings
from google.cloud.storage import Client as SClient
from google.oauth2 import service_account
import api.models as models

key = json.loads(base64.b64decode(os.environ['SISM_GOOGLE_CREDENTIALS']))
firebase_credentials = credentials.Certificate(key)
firebase_app = firebase_admin.initialize_app(firebase_credentials, {'projectId': 'songisms'})
storage_credentials = service_account.Credentials.from_service_account_info(key)
sclient = SClient(credentials=storage_credentials)
bucket = sclient.bucket(settings.GS_BUCKET_NAME)

vocab = 'en_core_web_sm'
try:
    nlp = spacy.load(vocab)
except:
    spacy.cli.download(vocab)
    nlp = spacy.load(vocab)

nlp.tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab)

inflector = inflect.engine()
syn_data = []
syn_file_name = './data/synonyms.txt'

with open(syn_file_name, 'r') as syn_file:
    syn_data = [
        [l.strip() for l in line.split(';')]
        for line in syn_file.readlines()
    ]


def tokenize_lyrics(val):
    val = val.lower()
    val = re.sub(r'[\,\"]', '', val)
    val = re.sub(r'\)', '', val)
    val = re.sub(r'\(', '', val)
    val = re.sub(r'\.+\b', ' ', val)
    val = re.sub(r'[^\w\s\'-.]', '', val)
    val = re.sub(r'(\w+in)\'[^\w]', r'\1g ', val)
    val = re.sub(r'\s+', ' ', val)
    toks = nlp(val)
    toks = [t.text for t in toks if t.pos_ not in ['PUNCT']]
    toks = [t + ('.' if '.' in t else '') for t in toks]
    # toks = [re.sub(r'\.+', '.', t) for t in toks]
    return toks


def get_lyric_ngrams(val, n_range=range(5)):
    ngrams = []
    toks = tokenize_lyrics(val)
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


def make_synonyms(word):
    syns = set()

    # in' <-> ing
    if len(word) >= 5:
        if word.endswith('in\''):
            syns.add(re.sub(r"'$", "g", word))
        elif word.endswith('ing'):
            syns.add(re.sub(r'g$', '\'', word))

    # add plural/singular if it only involves adding/removing an s
    simple_plural = word + 's'
    simple_sing = word[:-1]

    if simple_plural == inflector.plural(word):
        syns.add(simple_plural)
    elif simple_sing == inflector.singular_noun(word):
        syns.add(simple_sing)

    match_w = word.lower().strip()
    matches = [line for line in syn_data if match_w in line]
    for line in matches:
        for l in line:
            tok = l.lower().strip()
            if tok != match_w:
                syns.add(tok)

    return list(syns)


def get_firestore():
    return firestore.client()


def get_cloud_storage():
    return sclient, bucket


def get_storage_blob(fname):
    return bucket.blob(fname)


def fetch_audio(song, convert=False):
    yt_id = song.youtube_id
    print("***** fetching video", song.id, song.title, yt_id)

    video = pafy.new(f'https://www.youtube.com/watch?v={yt_id}')
    yt_meta = dict(
        id=yt_id,
        song_id=song.id,
        created=time.time(),
        author=video.author,
        bigthumb=video.bigthumb,
        bigthumbhd=video.bigthumbhd,
        category=video.category,
        dislikes=video.dislikes,
        duration=video.duration,
        expiry=video.expiry,
        likes=video.likes,
        thumb=video.thumb,
        title=video.title,
        viewcount=video.viewcount,
        watchv_url=video.watchv_url,
        # description=video.description,
        # keywords
    )

    md = song.metadata or dict()
    md['youtube'] = yt_meta
    song.metadata = md

    audio = video.getbestaudio()
    fname = f'{song.spotify_id}.{audio.extension}'
    tmpfile = f'/tmp/{fname}'

    if not os.path.exists(tmpfile):
        print('download', tmpfile)
        audio.download(filepath=tmpfile, quiet=False)
    else:
        print('webm file exists')

    if convert:
        fname_mp3 = f'{yt_id}.mp3'
        tmpfile_mp3 = f'/tmp/{fname_mp3}'
        if not os.path.exists(tmpfile_mp3):
            print('convert to mp3...')
            ffmpeg.input(tmpfile).output(tmpfile_mp3, ac=1, audio_bitrate='128k').run()
        else:
            print('mp3 exists')
        fname_upload = fname_mp3
        tmpfile_upload = tmpfile_mp3
    else:
        fname_upload = fname
        tmpfile_upload = tmpfile

    if os.path.exists(tmpfile_upload):
        print('uploading audio', tmpfile_upload)
        with open(tmpfile_upload, 'rb') as f:
            song.audio_file.save(fname_upload, File(f))
            song.save()
    else:
        print('missing upload file', tmpfile_upload)

    try:
        if os.path.exists(tmpfile):
            os.remove(tmpfile)
        if os.path.exists(tmpfile_mp3):
            os.remove(tmpfile_mp3)
    except:
        print("problem removing temp files", tmpfile, tmpfile_mp3)

    print("done")


def make_ngrams(song, force_update=False):
    ngrams = get_lyric_ngrams(song.lyrics, range(3))
    print('found', len(ngrams), 'ngrams...')

    with transaction.atomic():
        print('creating ngrams...')
        for key, n in ngrams:
            ngram, _ = models.NGram.objects.get_or_create(text=key, n=n)
            if not ngram.phones or force_update:
                ngram.phones = get_phones(ngram.text, vowels_only=True, include_stresses=False)
                # ngram.stresses = get_stresses(ngram.text)
                # ngram.ipa = get_ipa(ngram.text)
                # ngram.formants = get_formants(ngram.text)
                ngram.save()
            ngram.songs.add(song)


def set_lyrics_ipa(song):
    lines = song.lyrics.split('\n')
    ipa = '\n'.join([get_ipa(l, phones=False) for l in lines])
    song.lyrics_ipa = ipa
    song.save()


def prune():
    artists = models.Artist.objects.annotate(song_ct=Count('songs')).filter(song_ct=0)
    for a in artists:
        print("[PRUNE ARTIST]", a.pk, a.name)
        a.delete()

    writers = models.Writer.objects.annotate(song_ct=Count('songs')).filter(song_ct=0)
    for w in writers:
        print("[PRUNE WRITER]", w.pk, w.name)
        w.delete()

    ngrams = models.NGram.objects.annotate(song_ct=Count('songs'), rhyme_ct=Count('rhymes')).filter(
        song_ct=0, rhyme_ct=0)
    for n in ngrams:
        print("[PRUNE NGRAM]", n.pk, n.text)
        n.delete()


def phones_for_word(w):
    w = re.sub(r'in\'', 'ing', w)
    val = pron.phones_for_word(w)
    return val[0] if len(val) else ''


def pstresses(w):
    val = pron.stresses(w)
    return val[0] if len(val) else ''


def get_stresses(q):
    return ' '.join([pstresses(w) for w in q.split()])


def get_phones(q, vowels_only=False, include_stresses=False):
    words = q.split()
    phones = [phones_for_word(w) for w in words]
    phones = [p for p in phones if p]
    if len(phones) != len(words):
        return ''
    phones = ' '.join(phones)
    if vowels_only:
        phones = re.sub(r'[A-Z]+\b', '', phones)
    if not include_stresses:
        phones = re.sub(r'[\d]+', '', phones)
    phones = re.sub(r'\s+', ' ', phones)
    phones = phones.strip()
    return phones


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


def get_paginator(qs, page_size, page, paginated_type, **kwargs):
    p = Paginator(qs, page_size)
    try:
        page_obj = p.page(page)
    except PageNotAnInteger:
        page_obj = p.page(1)
    except EmptyPage:
        page_obj = p.page(p.num_pages)
    return paginated_type(
        page=page_obj.number,
        pages=p.num_pages,
        total=p.count,
        has_next=page_obj.has_next(),
        has_prev=page_obj.has_previous(),
        items=page_obj.object_list,
        **kwargs
    )


def GraphenePaginatedType(id, T):
    return type(id, (graphene.ObjectType,), dict(
        items=graphene.List(T),
        page=graphene.Int(),
        pages=graphene.Int(),
        total=graphene.Int(),
        has_next=graphene.Boolean(),
        has_prev=graphene.Boolean(),
        q=graphene.String(),
    ))


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
