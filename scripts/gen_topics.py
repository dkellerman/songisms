#!/usr/bin/env python ./manage.py script

import sys
import numpy as np
import gensim
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from api.models import Song
from api.nlp_utils import tokenize_lyrics
from biterm.utility import vec_to_biterms, topic_summuary
from biterm.btm import oBTM
from tqdm import tqdm


def gen_topics(num_topics=20):
    # load songs
    songs = Song.objects.exclude(lyrics=None).exclude(metadata__songMeanings__comments=None)
    stop_words = stopwords.words('english')
    data_words = []
    print('processing songs...')
    for song in tqdm(songs):
        toks = tokenize_lyrics(song.lyrics, stop_words)
        comments = song.metadata['songMeanings']['comments']
        for comment in comments:
            if is_english(comment['content']):
                toks += tokenize_lyrics(comment['content'].lower().strip(), stop_words + ['lyrics', 'song'])
        data_words.append(' '.join(toks))

    # import pdb; pdb.set_trace()
    vec = CountVectorizer(stop_words='english')
    X = vec.fit_transform(data_words).toarray()
    vocab = np.array(vec.get_feature_names())
    biterms = vec_to_biterms(X)
    btm = oBTM(num_topics=num_topics, V=vocab)
    # topics = btm.fit_transform(biterms, iterations=100)

    print("\n\n Train Online BTM ..")
    for i in range(0, len(biterms), 100):
        biterms_chunk = biterms[i:i + 100]
        btm.fit(biterms_chunk, iterations=50)
    topics = btm.transform(biterms)

    print("\n\n Visualize Topics ..")
    vis = pyLDAvis.prepare(btm.phi_wz.T, topics, np.count_nonzero(X, axis=1), vocab, np.sum(X, axis=0))
    pyLDAvis.save_html(vis, './data/online_btm_2.html')

    print("\n\n Topic coherence ..")
    topic_summuary(btm.phi_wz.T, X, vocab, 10)

    print("\n\n Texts & Topics ..")
    for i in range(len(data_words)):
        with open('./data/topics.txt', 'w') as f:
            f.write("{} (topic: {})".format(data_words[i], topics[i].argmax()))


    # gen lda model
    # id2word = corpora.Dictionary(data_words)
    # texts = data_words
    # corpus = [id2word.doc2bow(text) for text in texts]
    # lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
    #
    # # save
    # prepared = gensimvis.prepare(lda_model, corpus, id2word)
    # pyLDAvis.save_html(prepared, './data/ldavis_' + str(num_topics) + '.html')


def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


if __name__ == '__main__':
    args = sys.argv[3:]
    if len(args):
        gen_topics(int(args[0]))
    else:
        print('usage: specify num topics')
