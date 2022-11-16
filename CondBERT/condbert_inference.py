import os
import sys

def add_sys_path(p):
    p = os.path.abspath(p)
    print(p)
    if p not in sys.path:
        sys.path.append(p)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from importlib import reload

import condbert
reload(condbert)
from condbert import CondBertRewriter

import torch
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import pickle
from tqdm.auto import tqdm, trange

device = torch.device('cuda:0')

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

model = BertForMaskedLM.from_pretrained(model_name)

model.to(device);
vocab_root = 'vocab/'

with open(vocab_root + "negative-words.txt", "r") as f:
    s = f.readlines()
negative_words = list(map(lambda x: x[:-1], s))
with open(vocab_root + "toxic_words.txt", "r") as f:
    ss = f.readlines()
negative_words += list(map(lambda x: x[:-1], ss))

with open(vocab_root + "positive-words.txt", "r") as f:
    s = f.readlines()
positive_words = list(map(lambda x: x[:-1], s))

import pickle
with open(vocab_root + 'word2coef.pkl', 'rb') as f:
    word2coef = pickle.load(f)

token_toxicities = []
with open(vocab_root + 'token_toxicities.txt', 'r') as f:
    for line in f.readlines():
        token_toxicities.append(float(line))
token_toxicities = np.array(token_toxicities)
token_toxicities = np.maximum(0, np.log(1/(1/token_toxicities-1)))
for tok in ['.', ',', '-']:
    token_toxicities[tokenizer.encode(tok)][1] = 3
for tok in ['you']:
    token_toxicities[tokenizer.encode(tok)][1] = 0

reload(condbert)
from condbert import CondBertRewriter

editor = CondBertRewriter(
    model=model,
    tokenizer=tokenizer,
    device=device,
    neg_words=negative_words,
    pos_words=positive_words,
    word2coef=word2coef,
    token_toxicities=token_toxicities,
)

print(editor.translate('You are an idiot!', prnt=False))

editor = CondBertRewriter(
    model=model,
    tokenizer=tokenizer,
    device=device,
    neg_words=negative_words,
    pos_words=positive_words,
    word2coef=word2coef,
    token_toxicities=token_toxicities,
    predictor=None,
)

from multiword import masked_predictor
reload(masked_predictor)
from multiword.masked_predictor import MaskedPredictor

predictor = MaskedPredictor(model, tokenizer, max_len=250, device=device, label=0, contrast_penalty=0.0)
editor.predictor = predictor

def adjust_logits(logits, label):
    return logits - editor.token_toxicities * 3
predictor.logits_postprocessor = adjust_logits
print(editor.replacement_loop('You are an idiot!', verbose=False))

import choosers
reload(choosers)
from choosers import EmbeddingSimilarityChooser

predictor = MaskedPredictor(
    model, tokenizer, max_len=250, device=device, label=0, contrast_penalty=0.0, 
    confuse_bert_args=True
)
editor.predictor = predictor

def adjust_logits(logits, label=0):
    return logits - editor.token_toxicities * 10

predictor.logits_postprocessor = adjust_logits

cho = EmbeddingSimilarityChooser(sim_coef=100, tokenizer=tokenizer)

with open('../../data/test/test_10k_toxic', 'r') as inputs:
    lines = list(inputs.readlines())[:10]
    for i, line in enumerate(tqdm(lines)):
        inp = line.strip()
        out = editor.replacement_loop(inp, verbose=False, chooser=cho, n_top=10, n_tokens=(1,2,3), n_units=1)
        print(out)
