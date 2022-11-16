VOCAB_DIRNAME = 'vocabularies' 
neg_out_name = VOCAB_DIRNAME + '/negative-words.txt'
pos_out_name = VOCAB_DIRNAME + '/positive-words.txt'


from transformers import BertTokenizer
import numpy as np
import os
from helper import LR_classifier
from wordfiles_creation import writing_files
from collections import Counter
class vocabulary():
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)

    tox_corpus_path = '../../data/train/train_toxic'
    norm_corpus_path = '../../data/train/train_normal'

    if not os.path.exists(VOCAB_DIRNAME):
        os.makedirs(VOCAB_DIRNAME)
    c = Counter()

    for fn in [tox_corpus_path, norm_corpus_path]:
        with open(fn, 'r') as corpus:
            for line in corpus.readlines():
                for tok in line.strip().split():
                    c[tok] += 1

    corpus_tox,corpus_norm=LR_classifier.cleaning_corpus(c,tox_corpus_path,norm_corpus_path)
    writing_files()
    LR_classifier.logistic_Regression()



