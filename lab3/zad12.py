import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

twenty_train = fetch_20newsgroups(subset='all', shuffle=True)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)


def predict(model, train):
    scores = cross_val_score(model, train.data, train.target, cv=5, n_jobs=3)
    accuracy = np.round(np.mean(scores), 2)
    print(f'{model} has cross validation accuracy of {accuracy}, std={np.std(scores)}')


mnb_clf = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2), max_df=0.8)),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', MultinomialNB(alpha=1e-3))
])

sgd_clf = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2), max_df=0.8)),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', SGDClassifier(alpha=1e-3, max_iter=5, tol=None))
])

pa_sgd_clf = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2), max_df=0.8)),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', PassiveAggressiveClassifier()),
])

predict(mnb_clf, twenty_train)
predict(sgd_clf, twenty_train)
predict(pa_sgd_clf, twenty_train)
