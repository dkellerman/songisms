#!/usr/bin/env python ./manage.py script

'''
Cleans up GPT's lyrical IPA output and aligns it to the real lyrics. Writes out a corpus
to a JSON file.
'''

import re
import sys
import json
from tqdm import tqdm
from songs.models import Song
from songisms import utils


if __name__ == '__main__':
    args = sys.argv[3:]
    word2ipa = {}

    if len(args) == 1:
        songs = Song.objects.filter(title__iexact=args[0])
    else:
        songs = tqdm(Song.objects.all())

    for song in songs:
        lines = [utils.normalize_lyric(l) for l in song.lyrics.split('\n') if l.strip()]
        ipa_lines_gpt = [utils.normalize_ipa(l) for l in song.metadata['ipa'].split('\n') if l.strip()]
        text = ' '.join(lines)
        ipa_text_gpt = ' '.join(ipa_lines_gpt)
        words = text.split()
        ipa_words_gpt = ipa_text_gpt.split()

        # check alignment
        is_aligned = False
        while not is_aligned:
            ipa_words_canon = utils.get_ipa_words(text)
            aligned_ipa_words_gpt, aligned_ipa_words_canon, _ = utils.align_vals(ipa_words_gpt, ipa_words_canon)
            is_aligned = True

            # check for some specific issues, and set is_aligned to False if we need to re-align after
            for idx, word in enumerate(words):
                ipa = str(aligned_ipa_words_gpt[idx])
                # check dashes were handled correctly
                if '-' in word and ipa and ipa != '_' and "ˈ" not in ipa and "ˌ" not in ipa and "-" not in ipa:
                    words[idx:idx+1] = word.split('-')
                    text = ' '.join(words)
                    is_aligned = False
                    break
                else:
                    # check word/letter split handled correctly, e.g. p1 -> p 1
                    match = re.match(r'([a-zA-Z]+)(\d+)', word)
                    if match:
                        words[idx:idx+1] = [match.group(1), match.group(2)]
                        text = ' '.join(words)
                        is_aligned = False
                        break

        # make output word->ipa dict
        for idx, word in enumerate(words):
            val = str(aligned_ipa_words_gpt[idx])

            if not val or (val == '_'):
                val = str(aligned_ipa_words_canon[idx])
            if not val or (val == '_'):
                continue

            val = utils.normalize_ipa(val)
            cur_val = word2ipa.get(word, None)
            if cur_val == val:
                continue

            if cur_val:
                if utils.remove_stresses(val) == cur_val:
                    word2ipa[word] = val if len(val) > len(cur_val) else cur_val
            else:
                word2ipa[word] = val

    with open('./data/ipa_gpt.json', 'w') as f:
        f.write(json.dumps(word2ipa, indent=2, ensure_ascii=False))
