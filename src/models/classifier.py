import abc
import os

import fasttext
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from src.config.configs import tencent_embedding_path, self_train_test_data_path, single_character_tf_idf_config, \
    strong_attack_config, self_train_model_path, self_train_train_data_path, DeepModelConfig, deep_model_path, \
    full_word_tf_idf_config
from src.data.basic_functions import root_dir
from src.data.dataset import Sentences
from src.models.deep_model import SimpleCnn, SimpleRNN
from src.models.textcnn import TextCNN
from src.embedding.word_vector import WordVector
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd

class Classifier:
    @abc.abstractmethod
    def load_model(self):
        print("error")
        return self

    @abc.abstractmethod
    def train(self, x=None, y=None):
        print("error")
        return self

    @abc.abstractmethod
    def predict(self, texts):
        print("error")
        return

    def evaluate(self):
        from src.manipulate.importance_based import ImportanceBased
        from src.manipulate.rule_based import DeleteDirtyWordFoundByNGram

        datas = Sentences.read_test_data()
        pr = ImportanceBased([self], word_vector=WordVector(), attack_config=strong_attack_config)

        preds = pr.classifiers[0].predict(datas["sentence"].values.tolist())
        print(preds)
        score = accuracy_score(datas["label"].values, np.array(preds).round())
        print("untokenized score:", score)

        preds = pr.classifiers[0].predict(
            datas["sentence"].map(lambda x: ' '.join(pr.tokenize_text(x))).values.tolist())
        score = accuracy_score(datas["label"].values, np.array(preds).round())
        print("tokenize score:", score)

        dn = DeleteDirtyWordFoundByNGram()

        datas["sentence"] = datas["sentence"].map(lambda x: dn.replace(x))
        preds = FastTextClassifier().load_model().predict(datas["sentence"].values.tolist())
        score = accuracy_score(datas["label"].values, np.array(preds).round())
        print("remove dirty word:", score)


class TransformerClassifier(Classifier):
    def load_model(self):
        pass

    def __init__(self, model_name='clue/albert_chinese_small', max_len=50, batch_size=64, learning_rate=3e-5,
                 epochs=3):
        self.threshold = None
        self.model = None
        self.learner = None
        self.predictor = None
        self.threshold = None

        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict_label(self, data: list):
        prob = np.array(self.predict(list(data)))
        return self.prob_convert_to_label(prob)

    def prob_convert_to_label(self, prob):
        prob = np.array(prob)
        labels = (prob > self.threshold).astype(int)
        return labels

    def predict(self, data: list):
        return self.predictor.predict_proba(data)[:, 1]

    def set_threshold(self, x, y):
        pov_num = (np.array(y) == 1).sum()
        pov_prediction = np.array(self.predict(list(x)))
        self.threshold = np.sort(pov_prediction)[::-1][pov_num:pov_num + 2].mean()
        return self

    def save_prob_prediction_result(self, prob, label_name, save_path):
        pd.DataFrame(prob).to_csv(os.path.join(save_path, label_name+"prob"+str(self.threshold)), encoding="utf-8")

    def train(self, x=None, y=None):
        from ktrain import text
        import ktrain
        # only support binary classification
        full_length = len(y)
        pov_num = (np.array(y) == 1).sum()
        neg_num = full_length - pov_num

        t = text.Transformer(self.model_name, maxlen=self.max_len, class_names=["0", "1"])
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
        trn = t.preprocess_train(train_x, train_y.to_list())
        val = t.preprocess_test(test_x, test_y.to_list())

        model = t.get_classifier()
        learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=self.batch_size)
        # TODO: disable class_weight
        # TODO: add early top parameter into config
        learner.autofit(self.learning_rate, self.epochs, class_weight={0: pov_num, 1: neg_num}, early_stopping=4, reduce_on_plateau=2)

        self.learner = learner
        self.predictor = ktrain.get_predictor(learner.model, t)
        # TODO: lower number of x
        self.set_threshold(x, y)

        return self


class FastTextClassifier(Classifier):

    def __init__(self, model_path=os.path.join(root_dir(), "models", "mini.ftz")):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        model = fasttext.load_model(self.model_path)
        self.model = model
        return self

    def train(self, x=None, y=None):
        if x is not None:
            temp_path = (os.path.join(root_dir(), "data", "t_train.csv"), os.path.join(root_dir(), "data", "t_test.csv"))
            Sentences.save_fasttext_train_data(x, y, path=temp_path)
            train_path, test_path = temp_path
        else:
            train_path, test_path = self_train_train_data_path, self_train_test_data_path

        try:
            self.model = fasttext.train_supervised(train_path,
                                                   dim=200,
                                                   pretrainedVectors=tencent_embedding_path,
                                                   autotuneValidationFile=test_path,
                                                   autotuneDuration=1200)
        except Exception:
            Sentences().save_train_data()
            self.model = fasttext.train_supervised(train_path,
                                                   dim=200,
                                                   pretrainedVectors=tencent_embedding_path,
                                                   autotuneValidationFile=test_path,
                                                   autotuneDuration=3000)

        print(self.model.test(self_train_test_data_path))
        return self

    def save_model(self):
        self.model.save_model(self_train_model_path)

    def get_dirty_word_in_the_model(self, threshold=0.5):
        words = np.array(self.model.get_words())
        scores = np.array(self.predict(list(words)))
        scores_index = scores.argsort()[::-1]
        scores = scores[scores_index]
        words = words[scores_index]
        is_dirty = np.array(scores > threshold)
        return list(words[is_dirty])

    def _get_word_in_the_model(self, threshold=0.5, dirty=True):
        words = np.array(self.model.get_words())
        scores = np.array(self.predict(list(words)))

        if not dirty:
            scores = 1 - scores

        scores_index = scores.argsort()[::-1]
        scores = scores[scores_index]
        words = words[scores_index]
        is_dirty = np.array(scores > threshold)
        return list(words[is_dirty])

    def get_good_word_in_the_model(self, threshold=0.5):
        return self._get_word_in_the_model(threshold, False)

    @staticmethod
    def _modify_predict_result(predictions):
        def transform_label(text_label):
            return int(text_label[0][-1])

        labels = predictions[0]
        probs = predictions[1]

        modified_predictions = []
        for i in range(len(labels)):
            if transform_label(labels[i]) == 1:
                modified_predictions.append(probs[i][0])

            if transform_label(labels[i]) == 0:
                modified_predictions.append(1 - probs[i][0])

        return modified_predictions

    def predict(self, texts):
        return self._modify_predict_result(self.model.predict(texts))


class TFIDFClassifier(Classifier):

    def __init__(self, tf_idf_config=single_character_tf_idf_config, x=None, y=None):
        self.tf_idf_config = tf_idf_config
        self.x = x
        self.y = y
        self.vectorizer: TfidfVectorizer = None
        self.classifier = None

    @staticmethod
    def train_tf_idf_features(text, tf_idf_config):
        vectorizer = TfidfVectorizer(input='content', encoding='utf-8', decode_error='ignore', lowercase=False,
                                     **tf_idf_config)
        vectorizer.fit(text)
        return vectorizer

    def transform_text_to_vector(self, text):
        return self.vectorizer.transform(text)

    def predict(self, texts):
        return self.classifier.predict_proba(self.transform_text_to_vector(texts))[:, 1]

    def train(self, x=None, y=None):
        if x is None:
            x, y = self.x, self.y
        self.vectorizer = self.train_tf_idf_features(x, self.tf_idf_config)
        vectors = self.transform_text_to_vector(x)
        clf = LogisticRegression(random_state=0).fit(vectors, y)
        self.classifier = clf
        return self

    def load_model(self):
        pass


class DeepModel(Classifier):
    def __init__(self, word_vector: WordVector, tokenizer: callable, config: DeepModelConfig, model_creator):
        self.word_vector = word_vector
        self.tokenizer = tokenizer
        self.config = config
        self.model = None
        self.model_creator = model_creator

    def create_model(self):
        self.model = self.model_creator(self.config.input_length, self.word_vector.vector.vectors.shape[0], self.word_vector.vector.vectors.shape[1], embedding_matrix=self.word_vector.vector.vectors)
        self.model.compile('adam', 'mean_absolute_error', metrics=['accuracy'])

        return self

    def summary(self):
        a = self.model_creator(self.config.input_length, self.word_vector.vector.vectors.shape[0], self.word_vector.vector.vectors.shape[1], embedding_matrix=self.word_vector.vector.vectors)
        a.model.build((20, 256))
        a.model.summary()
        from keras.utils.vis_utils import plot_model
        plot_model(a.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    def get_embedding_and_middle_layers(self):
        return self.model.export_middle_layers()

    def save_model(self):
        self.model.save(deep_model_path)

    def load_model(self):
        self.model = tf.keras.models.load_model(deep_model_path)

    def train(self, x=None, y=None):

        self.create_model()

        index_data = self.text_to_id(x)
        train_data_i = tf.keras.preprocessing.sequence.pad_sequences(index_data, maxlen=self.config.input_length, dtype='int32',
                                                                     padding='post', truncating='pre', value=0)

        self.model.fit(x=train_data_i, y=y, batch_size=16, epochs=5, verbose=self.config.verbose, callbacks=None,
                       validation_split=0.1, validation_data=None, shuffle=True, class_weight=None,
                       sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None,
                       validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
        return self

    def predict(self, texts):
        index_data = self.text_to_id(texts)
        data = tf.keras.preprocessing.sequence.pad_sequences(index_data, maxlen=self.config.input_length, dtype='int32',
                                                      padding='post', truncating='pre', value=0)
        return self.model.predict(data).flatten()

    @staticmethod
    def map_func(word, word_dictionary):
        try:
            return word_dictionary[word]
        except KeyError:
            return 0

    def text_to_id(self, sentences):
        return self._transform_sentences_to_id(sentences, self.word_vector.vector.vocab.keys(), self.tokenizer)

    def get_dictionary(self):
        return dict(zip(self.word_vector.vector.vocab.keys(), range(len(self.word_vector.vector.vocab.keys()))))

    @staticmethod
    def _transform_sentences_to_id(sentences, word_list, tokenizer_function):

        word_dictionary = dict(zip(word_list, range(len(word_list))))

        result = []
        for sentence in sentences:
            tokenized_sentence = tokenizer_function(sentence)
            result.append(list(map(lambda x: DeepModel.map_func(x, word_dictionary), tokenized_sentence)))
        return result


class TFIDFEmbeddingClassifier(TFIDFClassifier):

    def __init__(self, word_vector: WordVector, tf_idf_config=single_character_tf_idf_config, x=None, y=None):
        super().__init__(tf_idf_config, x, y)
        self.word_vector = word_vector

    def transform_text_to_vector(self, text):
        # TODO: BUG warning fix default vector length
        final_vectors = []

        sparse_matrix = self.vectorizer.transform(text)
        values = sparse_matrix.data.tolist()
        features = self.vectorizer.inverse_transform(sparse_matrix)

        for word_feature in features:
            vectors = np.array([self.word_vector.get_vector(word) * values.pop(0) for word in word_feature])
            if len(word_feature) == 0:
                # print("word feature error, but fixed")
                final_vectors.append(np.zeros(len(self.word_vector.get_vector("你"))))
            else:
                final_vectors.append(np.mean(vectors, axis=0))

        return np.vstack(final_vectors)


class EmbeddingSVM(TFIDFEmbeddingClassifier):
    def train(self, x=None, y=None):
        if x is None:
            x, y = self.x, self.y

        self.vectorizer = self.train_tf_idf_features(x, self.tf_idf_config)
        vectors = self.transform_text_to_vector(x)
        clf = SVC(random_state=0, probability=True).fit(vectors, y)
        self.classifier = clf
        return self


class EmbeddingLGBM(EmbeddingSVM):

    def train(self, x=None, y=None):
        if x is None:
            x, y = self.x, self.y

        self.vectorizer = self.train_tf_idf_features(x, self.tf_idf_config)
        vectors = self.transform_text_to_vector(x)
        param = {'num_leaves': 64, 'objective': 'binary', 'lambda': 10,
         'subsample': 0.80, 'colsample_bytree': 0.75, 'min_child_weight': 3, 'eta': 0.02, 'seed': 0, 'verbose': -1,
         "gamma": 1}

        train_data = lgb.Dataset(vectors, label=y, params={'verbose': -1})
        num_round = 500

        bst = lgb.train(param, train_data, num_round, verbose_eval=False)

        self.classifier = bst
        return self

    def predict(self, texts):
        return self.classifier.predict(self.transform_text_to_vector(texts))


if __name__ == '__main__':
    from src.data.dataset import Tokenizer
    # data = Sentences.read_train_data()
    # EmbeddingLGBM(word_vector=WordVector(), tf_idf_config=full_word_tf_idf_config, x=data["sentence"].values, y=data["label"].values).train().evaluate()
    # DeepModel(word_vector=WordVector(),
    #           config=DeepModelConfig(),
    #           tokenizer=list,
    #           model_creator=SimpleRNN).train(x=data["sentence"].values, y=data["label"].values).evaluate()
    # # # print(score)
    # FastTextClassifier(self_train_model_path).evaluate()
    print(FastTextClassifier().get_good_word_in_the_model(threshold=0.5))
