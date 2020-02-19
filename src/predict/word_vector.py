import os

from gensim.models import KeyedVectors

from src.config.configs import tencent_embedding_path
from src.data.basic_functions import root_dir


class WordVector:
    def __init__(self, path=os.path.join(root_dir(), "models", "zh.300.vec.gz")):
        self.path = path
        self.vector = None
        self.load_model()

    def load_model(self):
        self.vector = KeyedVectors.load_word2vec_format(self.path, limit=50000)
        return self

    @staticmethod
    def _modify_similar_result(result):
        return [item[0] for item in result]

    def most_similar(self, text, topn=10):
        return self._modify_similar_result(self.vector.most_similar(text, topn=topn))


if __name__ == '__main__':
    word = "弱智"
    print(WordVector().load_model().most_similar(word))
    print(WordVector(path=tencent_embedding_path).load_model().most_similar(word))

