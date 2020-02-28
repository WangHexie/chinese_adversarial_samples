import os

from gensim.models import KeyedVectors

from src.config.configs import tencent_embedding_path
from src.data.basic_functions import root_dir
import numpy as np


class WordVector:
    def __init__(self, path=os.path.join(root_dir(), "models", "zh.300.vec.gz")):
        self.path = path
        self.vector = None
        self.load_model()

        self.word_count = dict()
        self.word_filter = []
        self.cache = dict()

    def _find_in_cache(self, word, topn):
        return self.cache[word+str(topn)]

    def _update_cache(self, word, topn, syns):
            self.cache[word+str(topn)] = syns

    def load_model(self):
        self.vector = KeyedVectors.load_word2vec_format(self.path, limit=50000)
        return self

    @staticmethod
    def _modify_similar_result(result):
        return [item[0] for item in result]

    def get_vector(self, word):
        try:
            return self.vector.get_vector(word)
        except KeyError:
            return np.zeros(len(self.get_vector("你")))

    def most_similar(self, text, topn=10):
        try:
            syns = self._find_in_cache(text, topn=topn)
        except KeyError:
            syns = self._modify_similar_result(self.vector.most_similar(text, topn=topn))
            self._update_cache(text, topn=topn, syns=syns)

        return syns

    def add_word_use(self, word, limitation):
        """
        limit number of use of a word to bypass new word finding model
        :param limitation: -1 : no limitation
        :param word:
        :return:
        """
        if word in self.word_count.keys():
            self.word_count[word] += 1
            if self.word_count[word] == limitation:
                self.word_filter.append(word)
        else:
            self.word_count[word] = 1

    def find_synonyms_with_word_count_and_limitation(self, word, topn):
        """

        :param word:
        :param topn:
        :param limitation: -1 means no limitation and act as same as most_similar function
        :return:
        """
        def plain_find(s, w, n):
            try:
                syns = s.most_similar(w, topn=n)
                syns = [syn_word for syn_word in syns if syn_word not in s.word_filter]
                s._update_cache(w, topn=n, syns=syns)
                return syns
            except KeyError:
                syns = []
                return syns

        origin_topn = topn
        topn_limit = 85  # use this parameter to avoid topn going to high
        synonyms = plain_find(self, word, topn)
        while (len(synonyms) < topn) and (topn < topn_limit):
            topn = 2*topn
            synonyms = plain_find(self, word, topn)

        return synonyms[:origin_topn]


if __name__ == '__main__':
    w = "剥皮"
    # print(WordVector().load_model().most_similar(w))
    print(WordVector(path=tencent_embedding_path).load_model().most_similar(w))

