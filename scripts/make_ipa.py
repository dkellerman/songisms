#!/usr/bin/env python ./manage.py script

'''Make data/ipa.csv file (requires espeak-ng to be installed)
'''

import time
from functools import lru_cache
from songs.models import Song
from songisms import utils
from tqdm import tqdm
from wonderwords import RandomWord
from espeakng import ESpeakNG

rw = RandomWord()
speaker = ESpeakNG()
speaker.voice = 'en-us'
got = 0
tot = 0
times = []
words = set()

def proc_ipa(tok):
    global got, tot, times, words
    t = time.time()
    ipa = espeak_ipa(tok)
    times.append(time.time() - t)
    if not ipa.strip():
        ipa = None
    words.add((tok, ipa))
    tot += 1
    if ipa:
        got += 1

@lru_cache(maxsize=None)
def espeak_ipa(tok):
    return speaker.g2p(tok, ipa=2)


if __name__ == '__main__':
    for s in tqdm(Song.objects.exclude(lyrics=None), 'songs'):
        for l in s.lyrics.split('\n'):
            if not l.strip():
                continue
            toks = utils.tokenize_lyric(l)
            for tok in toks:
                if not tok.strip():
                    continue
                proc_ipa(tok)

    for l in tqdm(utils.data.lines, 'lines'):
        toks = utils.tokenize_lyric(l)
        for tok in toks:
            if not tok.strip():
                continue
            proc_ipa(tok)

    for i in tqdm(range(1000), 'random'):
        word = utils.normalize_lyric(rw.word())
        if not word.strip():
            continue
        proc_ipa(word)

    common = [c for c in list(utils.data.get_common_words(10000).keys())
            if utils.remove_all_punctuation(c).strip()]
    for c in tqdm(common, 'common'):
        word = utils.normalize_lyric(c)
        proc_ipa(word)

    for _, r1, r2 in tqdm(utils.data.rhymes_train, 'train'):
        proc_ipa(r1)
        proc_ipa(r2)

    for _, r1, r2 in tqdm(utils.data.rhymes_test, 'test'):
        proc_ipa(r1)
        proc_ipa(r2)

    print(f"got {got} out of {tot} (missed {tot-got})",
          "({:.2f}%)".format(got / tot * 100))
    print(f"avg time {sum(times) / len(times) * 1000:.2f}ms")

    with open('./data/ipa.csv', 'w') as f:
        f.write("word,ipa\n")
        for tok, ipa in words:
            if utils.remove_non_lyric_punctuation(tok).strip() and ipa:
                f.write(f"{tok.lower()},{ipa}\n")

