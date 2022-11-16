VOCAB_DIRNAME = 'vocabularies' 
import pickle
import numpy as np
from condbert import CondBertRewriter
from choosers import EmbeddingSimilarityChooser
from multiword.masked_token_predictor_bert import MaskedTokenPredictorBert
import torch
import os
from transformers import BertTokenizer, BertForMaskedLM


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')
device = torch.device('cpu')

from vocab_creation import vocabulary
vocabulary()

with open(VOCAB_DIRNAME + "/negative-words.txt", "r") as f:
    s = f.readlines()
negative_words = list(map(lambda x: x[:-1], s))

with open(VOCAB_DIRNAME + "/positive-words.txt", "r") as f:
    s = f.readlines()
positive_words = list(map(lambda x: x[:-1], s))

with open(VOCAB_DIRNAME + '/word2coef.pkl', 'rb') as f:
    word2coef = pickle.load(f)

token_toxicities = []
with open(VOCAB_DIRNAME + '/token_toxicities.txt', 'r') as f:
    for line in f.readlines():
        token_toxicities.append(float(line))
token_toxicities = np.array(token_toxicities)
token_toxicities = np.maximum(0, np.log(1/(1/token_toxicities-1)))   # log odds ratio


def adjust_logits(logits, label=0):
    return logits - token_toxicities * 100 * (1 - 2 * label)

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)


predictor = MaskedTokenPredictorBert(model, tokenizer, max_len=250, device=device, label=0, contrast_penalty=0.0, logits_postprocessor=adjust_logits)

editor = CondBertRewriter(
    model=model,
    tokenizer=tokenizer,
    device=device,
    neg_words=negative_words,
    pos_words=positive_words,
    word2coef=word2coef,
    token_toxicities=token_toxicities,
    predictor=predictor,
)
chooser = EmbeddingSimilarityChooser(sim_coef=10, tokenizer=tokenizer)


