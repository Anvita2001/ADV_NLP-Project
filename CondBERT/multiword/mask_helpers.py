import numpy as np
from torch.utils.data import DataLoader
from keras_preprocessing.sequence import pad_sequences

import logging
logger = logging.getLogger('usem-experiments')

def find_bpe_position_by_offset(bpe_offsets, target_offset):
    bpe_nums=[]
    for sent_num, sent in enumerate(bpe_offsets):
        if sent[-1][0] < target_offset[0]:
            continue
        for bpe_num, bpe in enumerate(sent):
            if target_offset[0] <= bpe[0] and bpe[1] <= target_offset[1]:
                bpe_nums.append(bpe_num)
        return (sent_num, bpe_nums)
    
def generate_seq_indexes(indexes):
    if not indexes:
        yield []
        return
    for ind in indexes[0]:
        for seq in generate_seq_indexes(indexes[1:]):
            yield [ind] + seq

def bpe_tokenize(bpe_tokenizer, sentence):
    sent_bpe_tokens = []
    sent_bpe_offsets = []
    for token in sentence:
        token_bpes = bpe_tokenizer.tokenize(token.text)
        sent_bpe_offsets += [(token.begin, token.end) for _ in range(len(token_bpes))]
        sent_bpe_tokens += token_bpes
    return sent_bpe_tokens, sent_bpe_offsets

def nlargest_indexes(arr, n_top):
    arr_ids = np.argpartition(arr, -n_top)[-n_top:]
    return arr_ids[np.argsort(-arr[arr_ids])]

def remove_masked_token_subwords(masked_position, bpe_tokens, bpe_offsets):
    if len(masked_position[1]) > 1:
        indexes_to_del = masked_position[1][1:]
        del bpe_tokens[masked_position[0]][indexes_to_del[0] : indexes_to_del[-1] + 1], \
            bpe_offsets[masked_position[0]][indexes_to_del[0] : indexes_to_del[-1] + 1]
    masked_position = (masked_position[0], masked_position[1][0])
    return masked_position,  bpe_tokens, bpe_offsets

def merge_sorted_results(objects_left, scores_left, 
                         objects_right, scores_right, max_elems):   
    result_objects = []
    result_scores = []
    j = 0
    i = 0
    while True:
        if (len(result_scores) == max_elems):
            break
        if i == len(scores_left):
            result_objects += objects_right[j : j + max_elems - len(result_scores)]
            result_scores += scores_right[j : j + max_elems - len(result_scores)]
            break
        if j == len(scores_right):
            result_objects += objects_left[i : i + max_elems - len(result_scores)]
            result_scores += scores_left[i : i + max_elems - len(result_scores)]
            break
        if scores_left[i] > scores_right[j]:
            result_objects.append(objects_left[i])
            result_scores.append(scores_left[i])
            i += 1
        else:
            result_objects.append(objects_right[j])
            result_scores.append(scores_right[j])
            j += 1
    return result_objects, result_scores