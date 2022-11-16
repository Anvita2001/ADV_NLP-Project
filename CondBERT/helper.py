
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pickle

class LR_classifier:
    def __init__(self,c,norm_corpus,tox_corpus):
            self.c=c
            self.norm_corpus=norm_corpus
            self.tox_corpus=tox_corpus
    def cleaning_corpus(c,norm_corpus,tox_corpus):
        vocab = {w for w, _ in c.most_common() if _ > 0} 
        for line in tox_corpus.readlines():
            for w in line.strip().split():
                if w in vocab:
                    corpus_tox=[' '.join[w]]
        for line in norm_corpus.readlines():
            for w in line.strip().split():
                if w in vocab:
                    corpus_norm=[' '.join[w]]
        return corpus_tox,corpus_norm

    def logistic_Regression(corpus_tox,corpus_norm):
        pipe = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=1000))
        X_train = corpus_tox + corpus_norm
        y_train = [1] * len(corpus_tox) + [0] * len(corpus_norm)
        pipe.fit(X_train, y_train)
        coefs = pipe[1].coef_[0]
        coefs.shape
        word2coef = {w: coefs[idx] for w, idx in pipe[0].vocabulary_.items()}
        with open('vocabularies/word2coef.pkl', 'wb') as f:
            pickle.dump(word2coef, f)

import numpy as np

class NgramSalienceCalculator():
    def __init__(self, tox_corpus, norm_corpus, use_ngrams=False):
        ngrams = (1, 3) if use_ngrams else (1, 1)
        self.vectorizer = CountVectorizer(ngram_range=ngrams)

        tox_count_matrix = self.vectorizer.fit_transform(tox_corpus)
        self.tox_vocab = self.vectorizer.vocabulary_
        self.tox_counts = np.sum(tox_count_matrix, axis=0)

        norm_count_matrix = self.vectorizer.fit_transform(norm_corpus)
        self.norm_vocab = self.vectorizer.vocabulary_
        self.norm_counts = np.sum(norm_count_matrix, axis=0)

    def salience(self, feature, attribute='tox', lmbda=0.5):
        assert attribute in ['tox', 'norm']
        if feature not in self.tox_vocab:
            tox_count = 0.0
        else:
            tox_count = self.tox_counts[0, self.tox_vocab[feature]]

        if feature not in self.norm_vocab:
            norm_count = 0.0
        else:
            norm_count = self.norm_counts[0, self.norm_vocab[feature]]

        if attribute == 'tox':
            return (tox_count + lmbda) / (norm_count + lmbda)
        else:
            return (norm_count + lmbda) / (tox_count + lmbda)
            
from collections import defaultdict
toxic_counter = defaultdict(lambda: 1)
nontoxic_counter = defaultdict(lambda: 1)
from transformers import BertTokenizer

class token_toxicity_calc():
    def __init__(self,corpus_tox,corpus_norm):
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        for text in (corpus_tox):
            for token in tokenizer.encode(text):
                toxic_counter[token] += 1
        for text in (corpus_norm):
            for token in tokenizer.encode(text):
                nontoxic_counter[token] += 1

        token_toxicities = [toxic_counter[i] / (nontoxic_counter[i] + toxic_counter[i]) for i in range(len(tokenizer.vocab))]
        with open('vocabularies/token_toxicities.txt', 'w') as f:
            for t in token_toxicities:
                f.write(str(t))
                f.write('\n')
