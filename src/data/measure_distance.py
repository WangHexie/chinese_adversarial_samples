import os

import editdistance
import numpy as np
from gensim.models import KeyedVectors

from src.data.basic_functions import root_dir


def normalized_levenshtein(str_a, str_b):
    '''
    Edit distance normalized to [0, 1].
    '''
    return min(editdistance.eval(str_a, str_b) / (len(str_b) + 1e-16), 1.0)


def jaccard_set(set_a, set_b):
    '''
    Jaccard SIMILARITY between sets.
    '''
    set_c = set_a.intersection(set_b)
    return float(len(set_c)) / (len(set_a) + len(set_b) - len(set_c) + 1e-16)


def jaccard_char(str_a, str_b):
    '''
    Jaccard DISTANCE between strings, evaluated by characters.
    '''
    set_a = set(str_a)
    set_b = set(str_b)
    return 1.0 - jaccard_set(set_a, set_b)


def jaccard_word(str_a, str_b, sep=' '):
    '''
    Jaccard DISTANCE between strings, evaluated by words.
    '''
    set_a = set(str_a.split(sep))
    set_b = set(str_b.split(sep))
    return 1.0 - jaccard_set(set_a, set_b)


def tokenize(text):
    import jieba
    return ' '.join(jieba.cut(text))


class DistanceCalculator:
    '''
    Computes pair-wise distances between texts, using multiple metrics.
    '''

    def __init__(self):
        self.EMBEDDING_PATH = os.path.join(root_dir(), "models", "zh.300.vec.gz")
        self.EMBEDDING_DIM = 300
        self.DEFAULT_KEYVEC = KeyedVectors.load_word2vec_format(self.EMBEDDING_PATH, limit=50000)

    def __doc2vec(self, tokenized):
        tokens = tokenized.split(' ')
        vec = np.full(self.EMBEDDING_DIM, 1e-10)
        weight = 1e-8
        for _token in tokens:
            try:
                vec += self.DEFAULT_KEYVEC.get_vector(_token)
                weight += 1.0
            except:
                pass
        return vec / weight

    def __batch_doc2vec(self, list_of_tokenized_text):
        return [self.__doc2vec(_text) for _text in list_of_tokenized_text]

    def __call__(self, docs_a, docs_b):
        docs_a_cut = [tokenize(_doc) for _doc in docs_a]
        docs_b_cut = [tokenize(_doc) for _doc in docs_b]

        # further validating input
        if not self.validate_input(docs_a, docs_b):
            raise ValueError("distance module got invalid input")

        # actual processing
        num_elements = len(docs_a)
        distances = dict()
        distances['normalized_levenshtein'] = np.array([normalized_levenshtein(docs_a[i], docs_b[i]) for i in
                                                        range(num_elements)])
        distances['jaccard_word'] = np.array([jaccard_word(docs_a_cut[i], docs_b_cut[i]) for i in range(num_elements)])
        distances['jaccard_char'] = np.array([jaccard_char(docs_a[i], docs_b[i]) for i in range(num_elements)])
        distances['embedding_cosine'] = np.array(self.batch_embedding_cosine_distance(docs_a_cut, docs_b_cut))
        distances['final_similarity_score'] = 1 - (distances['normalized_levenshtein'] * 3 / 14 + \
                                                   distances['jaccard_word'] * 1 / 7 + \
                                                   distances['jaccard_char'] * 3 / 14 + \
                                                   distances['embedding_cosine'] * 3 / 7)
        return distances

    def validate_input(self, text_list_a, text_list_b):
        '''
        Determine whether two arguments are lists containing the same number of strings.
        '''
        if not (isinstance(text_list_a, list) and isinstance(text_list_b, list)):
            return False

        if not len(text_list_a) == len(text_list_b):
            return False

        for i in range(len(text_list_a)):
            if not (isinstance(text_list_a[i], str) and isinstance(text_list_b[i], str)):
                return False

        return True

    def batch_embedding_cosine_distance(self, text_list_a, text_list_b):
        '''
        Compute embedding cosine distances in batches.
        '''
        embedding_array_a = np.array(self.__batch_doc2vec(text_list_a))
        embedding_array_b = np.array(self.__batch_doc2vec(text_list_b))
        norm_a = np.linalg.norm(embedding_array_a, axis=1)
        norm_b = np.linalg.norm(embedding_array_b, axis=1)
        cosine_numer = np.multiply(embedding_array_a, embedding_array_b).sum(axis=1)
        cosine_denom = np.multiply(norm_a, norm_b)
        cosine_dist = 1.0 - np.divide(cosine_numer, cosine_denom)
        return cosine_dist.tolist()


if __name__ == '__main__':
    print(DistanceCalculator()(["你到底说啥"], ["你到底说啥了"]))
