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
import pronouncing as pron

rw = RandomWord()
speaker = ESpeakNG()
speaker.voice = 'en-us'
got = 0
tot = 0
times = []
words = dict()

def handle(tok):
    global got, tot, times, words
    tok = utils.normalize_lyric(tok)
    if words.get(tok) or not utils.remove_all_punctuation(tok):
        return
    t = time.time()
    ipa = espeak_ipa(tok)
    times.append(time.time() - t)
    if not ipa.strip():
        ipa = None
    words[tok] = ipa
    tot += 1
    if ipa:
        got += 1

@lru_cache(maxsize=None)
def espeak_ipa(tok):
    return speaker.g2p(tok, ipa=2)


if __name__ == '__main__':
    for s in tqdm(Song.objects.exclude(lyrics=None), 'songs'):
        for tok in utils.tokenize_lyric(s.lyrics):
            handle(tok)
        if s.rhymes_raw:
            for tok1, tok2 in utils.get_rhyme_pairs(s.rhymes_raw):
                handle(tok1)
                handle(tok2)

    for l in tqdm(utils.data.lines, 'lines'):
        toks = utils.tokenize_lyric(l)
        for tok in toks:
            handle(tok)

    for i in tqdm(range(1000), 'random'):
        word = rw.word()
        handle(word)

    common = [c for c in list(utils.data.get_common_words(10000).keys())]
    for c in tqdm(common, 'common'):
        handle(word)

    for _, r1, r2 in tqdm(utils.data.rhymes_train, 'train'):
        handle(r1)
        handle(r2)

    for _, r1, r2 in tqdm(utils.data.rhymes_test, 'test'):
        handle(r1)
        handle(r2)

    words2 = list(words.keys())
    for tok in tqdm(words2, 'rhymes'):
        for rtok in [r for r in pron.rhymes(tok) if r in common]:
            handle(rtok)

    print(f"got {got} out of {tot} (missed {tot-got})",
          "({:.2f}%)".format(got / tot * 100))
    print(f"avg time {sum(times) / len(times) * 1000:.2f}ms")

    with open('./data/ipa.csv', 'w') as f:
        f.write("word,ipa\n")
        for tok, ipa in words.items():
            if utils.remove_non_lyric_punctuation(tok).strip() and ipa:
                f.write(f"{tok.lower()},{ipa}\n")

