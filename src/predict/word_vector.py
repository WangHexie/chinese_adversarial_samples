import os

from gensim.models import KeyedVectors

from src.config.configs import tencent_embedding_path
from src.data.basic_functions import root_dir


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
        :param limitation:
        :param word:
        :return:
        """
        if word in self.word_count.keys():
            self.word_count[word] += 1
            if self.word_count[word] == limitation:
                self.word_filter.append(word)
        else:
            self.word_count[word] = 1

    def find_synonyms_with_word_count_and_limitation(self, word, topn, limitation):
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

        if limitation == -1:
            try:
                return self.most_similar(word, topn)
            except KeyError:
                return []

        limit = limitation
        topn_limit = 200
        synonyms = plain_find(self, word, topn)
        while (len(synonyms) < limit) and (topn < topn_limit):
            topn = 2*topn
            synonyms = plain_find(self, word, topn)

        return synonyms[:limit]


if __name__ == '__main__':
    w = "妳"
    print(WordVector().load_model().most_similar(w))
    print(WordVector(path=tencent_embedding_path).load_model().most_similar(w))

