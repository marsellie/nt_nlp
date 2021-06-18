from re import sub
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def prepare_data(filename):
    n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']
    result = pd.read_csv(
        filename,
        sep=';',
        error_bad_lines=False,
        names=n,
        usecols=['text'],
        encoding='utf-8',
        quoting=3)
    return result.dropna().drop_duplicates()


data_positive = prepare_data('positive.csv')
data_negative = prepare_data('negative.csv')

sample_size = min(data_positive.shape[0],
                  data_negative.shape[0])

raw_data = np.concatenate((data_positive['text'].values[:sample_size],
                           data_negative['text'].values[:sample_size]), axis=0)
labels = [1] * sample_size + [0] * sample_size


def preprocess_text(text):
    text = text.lower().replace("ё", "е")
    text = sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    text = sub('@[^\s]+', 'USER', text)
    text = sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = sub(' +', ' ', text)
    return text.strip()


data = [preprocess_text(str(t)) for t in raw_data]
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=2)

classifiers = [
    {'classifier': LogisticRegression, 'name': 'logistic'},
    {'classifier': LinearSVC, 'name': 'linear svc'},
    {'classifier': KNeighborsClassifier, 'name': 'kneig'},
    {'classifier': DecisionTreeClassifier, 'name': 'decision tree'},
]

vectorizers = [
    {'vectorizer': HashingVectorizer, 'name': 'hash'},
    {'vectorizer': CountVectorizer, 'name': 'count'},
    {'vectorizer': TfidfVectorizer, 'name': 'tfidf'},
]

for clasz in classifiers:
    for vector in vectorizers:
        vectorizer = vector['vectorizer'](decode_error='ignore')

        vectorizer.fit_transform([str(x) for x in data])

        classifier = clasz['classifier']()
        classifier.fit(vectorizer.transform(x_train), y_train)

        y_predicted = classifier.predict(vectorizer.transform(x_test))
        report = classification_report(y_test, y_predicted, digits=6)
        print('====', clasz['name'], ' : ', vector['name'], '====')
        print(report)
