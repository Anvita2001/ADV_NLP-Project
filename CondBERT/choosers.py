import numpy as np

from scipy.spatial.distance import cosine as cs


class EmbeddingSimilarityChooser:
    def __init__(self, sim_coef=100, tokenizer=None):
        self.glove_embedding = flair.embeddings.WordEmbeddings('glove')
        self.sim_coef = sim_coef
        self.tokenizer = tokenizer

    def embed(self, text):
        toks = self.glove_embedding.embed(flair.data.Sentence(text))[0]
        if not toks:
            return np.zeros(self.glove_embedding.embedding_length)
        return np.mean([t.embedding.cpu().numpy() for t in toks], axis=0)
    
    def decode(self, tokens):
        if isinstance(tokens, str):
            return tokens
        if self.tokenizer:
            return self.tokenizer.convert_tokens_to_string(tokens)
        return ' '.join(tokens).replace(' ##', '')

    def __call__(self, hypotheses, original=None, scores=None, **kwargs):
        e = self.embed(self.decode(original))
        canditates = []
        for word,score in zip(hypotheses,scores):
            canditates.append((word, score, cs(e, self.embed(self.decode(word)))))
        candidates = sorted(candidates, key=lambda x: x[1] + x[2] * self.sim_coef, reverse=True)
        return candidates[0][0]
