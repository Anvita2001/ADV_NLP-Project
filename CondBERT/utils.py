from collections import defaultdict

def group_by_first_token(texts, tokenizer):
    seqs = [tokenizer.encode(x, add_special_tokens=False) for x in texts]
    grouped = defaultdict(list)
    for seq in seqs:
        grouped[seq[0]].append(seq)
    return grouped

def default_chooser(hypotheses, original=None, **kwargs):
    return hypotheses[0]

def convert_mask(tokenizer, tok_ids, mask_ids, duplicate=False, start_from=0):
    toks_tmp = [tokenizer.convert_ids_to_tokens(tok_ids[0])[1:-1]]
    mask_pos = None
    toks = []
    mask_toks = []
    has_mask = False
    for i, is_masked in enumerate(mask_ids[0][1:-1]):
        tok = toks_tmp[0][i]
        if not has_mask:
            if is_masked and i >= start_from and not tok.startswith('##'):
                has_mask = True
                mask_pos = [i]
                mask_toks.append(tok)
            toks.append(tok)
        else:
            if not is_masked or not tok.startswith('##'):
                toks.extend(toks_tmp[0][i:])
                break
            else:
                mask_toks.append(tok)
    toks = [toks]

    if duplicate:
        toks = [toks_tmp[0] + ['[SEP]'] + toks[0]]
        mask_pos[0] += len(toks_tmp[0]) + 1
    return toks, mask_pos, mask_toks

def toks_to_words(v, token_ids):
    indices = []
    for i, token_id in enumerate(token_ids):
        token_text = v[token_id]
        if token_text.startswith('##'):
            indices.append(i)
        else:
            if indices:
                toks = [v[token_ids[t]] for t in indices]
                word = ''.join([toks[0]] + [t[2:] for t in toks[1:]])
                yield indices, word
            indices = [i]

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
