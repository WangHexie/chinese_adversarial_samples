from dataclasses import asdict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate

from src.config.configs import TFIDFConfig, full_word_tf_idf_config, DeepModelConfig
from src.data.dataset import Sentences, Tokenizer
from src.embedding.word_vector import WordVector
from src.models.classifier import TFIDFClassifier, FastTextClassifier, EmbeddingSVM, EmbeddingLGBM, DeepModel
from src.models.deep_model import SimpleCnn, SimpleRNN


class ClassifierWrapper(BaseEstimator):
    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, x, y):
        self.classifier.train(x, y)
        return self

    def predict(self, x):
        return (np.array(self.classifier.predict(x)) > 0.5).astype(int).tolist()


def evaluate_classifier(classifier):
    data = Sentences.read_full_data()
    data = data.sample(frac=1)
    print(data.shape)
    x, y = data["sentence"].values, data["label"].values
    clf = ClassifierWrapper(classifier)
    scores = cross_validate(clf, x, y, cv=5, scoring=['precision_macro', 'recall_macro', "f1"])
    print(scores)
    for key in scores.keys():
        print("{key}: {value}".format(key=key, value=scores[key].mean()))


def evaluate_all():
    word_vector = WordVector()
    classifiers = {
        "rnn_tokenizer": DeepModel(word_vector=word_vector, config=DeepModelConfig(), tokenizer=Tokenizer().tokenize,
                                   model_creator=SimpleRNN),

        "embedding svn": EmbeddingSVM(word_vector=word_vector),  # TODO : bug. prob output is clearly wrong
        "tfidf tokenizer": TFIDFClassifier(tf_idf_config=full_word_tf_idf_config),
        "tfidf ngram": TFIDFClassifier(tf_idf_config=asdict(TFIDFConfig(ngram_range=(1, 3), min_df=5))),

        "embedding lgbm": EmbeddingLGBM(word_vector=word_vector),
        # "fasttext self-trained": FastTextClassifier(),
        "cnn": DeepModel(word_vector=word_vector, config=DeepModelConfig(), tokenizer=list, model_creator=SimpleCnn),
        "cnn_tokenizer": DeepModel(word_vector=word_vector, config=DeepModelConfig(), tokenizer=Tokenizer().tokenize, model_creator=SimpleCnn)
    }
    for key in classifiers.keys():
        print(key)
        evaluate_classifier(classifiers[key])


if __name__ == '__main__':
    # import warnings
    #
    # warnings.simplefilter("ignore")
    # evaluate_all()
    word_vector = WordVector()
    a = DeepModel(word_vector=word_vector, config=DeepModelConfig(), tokenizer=Tokenizer().tokenize,
              model_creator=SimpleCnn)
    a.summary()
    # a.model.build((20, 256))
    # a.model.summary()
    # from keras.utils.vis_utils import plot_model
    # plot_model(a.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
