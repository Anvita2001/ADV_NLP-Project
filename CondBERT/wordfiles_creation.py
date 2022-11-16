from helper import NgramSalienceCalculator

class writing_files():
        def __init__(self,corpus_tox, corpus_norm):
            sc = NgramSalienceCalculator(corpus_tox, corpus_norm, False)
            seen_grams = set()
            with open('vocabularies/negative-words.txt', 'w') as neg_out, open('vocabularies/positive-words.txt', 'w') as pos_out:
                for gram in set(sc.tox_vocab.keys()).union(set(sc.norm_vocab.keys())):
                    if gram not in seen_grams:
                        seen_grams.add(gram)
                        toxic_salience = sc.salience(gram, attribute='tox')
                        polite_salience = sc.salience(gram, attribute='norm')
                        if toxic_salience > 4:
                            neg_out.writelines(f'{gram}\n')
                        elif polite_salience > 4:
                            pos_out.writelines(f'{gram}\n')
