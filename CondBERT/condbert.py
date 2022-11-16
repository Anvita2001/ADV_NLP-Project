import torch
from utils import *

class CondBert:
    def __init__(self,model,tokenizer,device,neg_words,pos_words,word2coef,token_toxicities,predictor=None,):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.neg_words = neg_words
        self.pos_words = pos_words
        self.word2coef = word2coef
        self.token_toxicities = token_toxicities
        self.predictor = predictor

        # calculated properties
        self.v = {v: k for k, v in tokenizer.vocab.items()}
        self.device_toxicities = torch.tensor(token_toxicities).to(self.device)

        self.neg_complex_tokens = group_by_first_token(neg_words, self.tokenizer)
        self.pos_complex_tokens = group_by_first_token(pos_words, self.tokenizer)
        self.mask_index = self.tokenizer.convert_tokens_to_ids("[MASK]")

    def get_mask_fast(self,inp: str,bad_words=None,min_bad_score=0,aggressive=True,max_score_margin=0.5,label=0):
        if bad_words is None:
            if label == 0:
                bad_words = self.neg_complex_tokens
            else:
                bad_words = self.pos_complex_tokens

        sentences = [self.tokenizer.encode(inp, add_special_tokens=True)]
        sentences_torch = torch.tensor(sentences)
        masks = torch.zeros_like(sentences_torch)

        for sent_id, sent in enumerate(sentences):
            for first_tok_id, tok in enumerate(sent):
                for hypothesis in bad_words.get(tok, []):
                    n = len(hypothesis)
                    if sent[first_tok_id: (first_tok_id + n)] == hypothesis:
                        for step in range(n):
                            masks[sent_id, first_tok_id + step] = 1
                        # if a word has toxic prefix, it is all toxic, so we should label its suffix as well
                        for offset, next_token in enumerate(sent[(first_tok_id + n):]):
                            if self.tokenizer.convert_ids_to_tokens(next_token).startswith('##'):
                                masks[sent_id, first_tok_id + n + offset] = 1
                            else:
                                break
            if sum(masks[sent_id].numpy()) == 0 or aggressive:
                scored_words = []
                for indices, word in self.toks_to_words(self.v,sent):
                    score = self.word2coef.get(word, 0) * (1 - 2 * label)
                    if score:
                        scored_words.append([indices, word, score])
                if scored_words:
                    max_score = max(s[2] for s in scored_words)
                    if max_score > min_bad_score:
                        for indices, word, score in scored_words:
                            if score >= max(min_bad_score, max_score * max_score_margin):
                                masks[sent_id, indices] = 1

        return sentences_torch, masks

    def replacement_loop(self,text,verbose=True,chooser=default_chooser,n_tokens=(1, 2, 3),n_top=10,mask_token=False,max_steps=1000,label=0,**predictor_args):
        span_detector = self.get_mask_fast
        predictor = self.predictor
        new_text = text
        look_from = 0

        for i in range(max_steps):
            tok_ids, mask_ids = span_detector(new_text, label=label)
            if not sum(mask_ids[0][(1 + look_from):]):
                break
            toks, mask_pos, mask_toks = self.convert_mask(
                self.tokenizer,tok_ids, mask_ids, duplicate=False, start_from=look_from
            )
            if mask_pos is None:
                return new_text
            texts, scores = predictor.generate(toks,mask_pos,n_tokens=list(n_tokens),n_top=n_top,fix_multiunit=False,mask_token=mask_token,label=label,**predictor_args)
            old_replacement = chooser(hypotheses=texts[0], scores=scores[0], original=mask_toks)
            if isinstance(old_replacement, str):
                old_replacement = [old_replacement]
            replacement = [t for w in old_replacement for t in w.split('_')]
            if verbose:
                print(mask_toks, '->', replacement)
            new_toks = toks[0][:mask_pos[0]] + replacement + toks[0][mask_pos[0] + 1:]
            new_text = self.tokenizer.convert_tokens_to_string(new_toks)
            look_from = mask_pos[0] + len(old_replacement)
        return new_text
    
    def translate(self,ss,prnt=True,toxicity_penalty=15):
        get_mask = self.get_mask_fast
        input_ids, attn_mask = get_mask(ss, bad_words=self.neg_complex_tokens, label=0)
        if attn_mask.sum().numpy() == 0:
            return ss

        masked = torch.ones_like(input_ids) * -100
        for i in range(input_ids.shape[0]):
            masked[i][attn_mask[i] == 1] = input_ids[i][attn_mask[i] == 1]

        input_ids = input_ids.to(self.device)

        self.model.eval()

        outputs = self.model(input_ids,token_type_ids=0)
        
        for i in range(input_ids.shape[0]):
            logits = outputs[-1][i][attn_mask[i] == 1]
            if toxicity_penalty:
                logits -= self.device_toxicities * toxicity_penalty
            else:
                scores = logits
            input_ids[i][attn_mask[i] == 1] = scores.argmax(dim=1)

        result = self.tokenizer.convert_tokens_to_string(
            [self.tokenizer.convert_ids_to_tokens(i.item()) for i in input_ids[0][1:-1]]
        )
        return result.split('[SEP] [CLS] ')[-1]

    