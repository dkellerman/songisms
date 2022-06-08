#!/usr/bin/env python

import django, os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "songisms.settings")
django.setup()

from api.models import *
from api.utils import *
from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn import metrics, naive_bayes


stop_words = stopwords.words('english')
data = []

print('loading data...')
f = 'writers'

for song in tqdm(Song.objects.prefetch_related(f).exclude(lyrics=None)):
    lyr = ' '.join(tokenize_lyrics(song.lyrics, stop_words=[], unique=True))
    objs = getattr(song, f).all()
    # for w in song.writers.all():
    if objs.count():
        obj = objs[0]
        val = (obj.name.lower(), lyr,)
        data.append(val)

corpus = [d[1] for d in data]
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(corpus)

labels = [d[0] for d in data]
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42
)

print('training...')
# svm = SVC(kernel='linear')
# svm.fit(X_train, y_train)

# fit the training dataset on the NB classifier
naive = naive_bayes.MultinomialNB()
naive.fit(X_train, y_train)

print('predicting...')
# y_pred = svm.predict(X_test)
y_pred = naive.predict(X_test)
print("Naive Bayes Accuracy Score -> ", metrics.accuracy_score(y_pred, y_test)*100)

#print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
#print(svm.score(X_test, y_test))
#print(confusion_matrix(y_pred, y_test))
# import pdb; pdb.set_trace()

