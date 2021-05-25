from re import sub
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


def save(model: LogisticRegression, tfidf: TfidfVectorizer):
    pickle.dump(model, open('model.sav', 'wb'))
    pickle.dump(tfidf, open('tfidf.sav', 'wb'))


def load() -> Tuple[LogisticRegression, TfidfVectorizer]:
    return pickle.load(open('model.sav', 'rb')), pickle.load(open('tfidf.sav', 'rb'))


def prepare_model() -> Tuple[LogisticRegression, TfidfVectorizer]:
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
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=2)

    tfidf = TfidfVectorizer(decode_error='ignore')
    tfidf.fit_transform([str(x) for x in data])

    classifier = LogisticRegression()
    classifier.fit(tfidf.transform(x_train), y_train)
    return classifier, tfidf
