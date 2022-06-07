#!/usr/bin/env python

import os, django, sys
import gensim
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from nltk.corpus import stopwords

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "songisms.settings")
django.setup()

from api.models import Song
from api.utils import tokenize_lyrics


def gen_topics(num_topics=10):
    # load songs
    songs = Song.objects.exclude(lyrics=None)
    stop_words = stopwords.words('english')
    data_words = [
        list(set([
            tok for tok in tokenize_lyrics(' '.join(song.lyrics.split('\n')))
            if tok not in stop_words
        ]))
        for song in songs
    ]

    # gen lda model
    id2word = corpora.Dictionary(data_words)
    texts = data_words
    corpus = [id2word.doc2bow(text) for text in texts]
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)

    # save
    prepared = gensimvis.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(prepared, './data/ldavis_' + str(num_topics) + '.html')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        gen_topics(int(sys.argv[1]))
    else:
        print('specify num topics')
