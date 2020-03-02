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
    strong_attack_config, self_train_model_path, self_train_train_data_path, DeepModelConfig, deep_model_path
from src.data.basic_functions import root_dir
from src.data.dataset import Sentences
from src.models.textcnn import TextCNN
from src.predict.word_vector import WordVector


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
        try:
            self.model = fasttext.train_supervised(self_train_train_data_path,
                                                   dim=200,
                                                   pretrainedVectors=tencent_embedding_path,
                                                   autotuneValidationFile=self_train_test_data_path,
                                                   autotuneDuration=1200)
        except Exception:
            Sentences().save_train_data()
            self.model = fasttext.train_supervised(self_train_train_data_path,
                                                   dim=200,
                                                   pretrainedVectors=tencent_embedding_path,
                                                   autotuneValidationFile=self_train_test_data_path,
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
        self.vectorizer = self.train_tf_idf_features(self.x, self.tf_idf_config)
        vectors = self.transform_text_to_vector(self.x)
        clf = LogisticRegression(random_state=0).fit(vectors, self.y)
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
        self.model.fit(x=train_data_i, y=y, batch_size=16, epochs=5, verbose=1, callbacks=None,
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
                print("word feature error, but fixed")
                final_vectors.append(np.zeros(len(self.word_vector.get_vector("你"))))
            else:
                final_vectors.append(np.mean(vectors, axis=0))

        return np.vstack(final_vectors)


class EmbeddingSVM(TFIDFEmbeddingClassifier):
    def train(self, x=None, y=None):
        self.vectorizer = self.train_tf_idf_features(self.x, self.tf_idf_config)
        vectors = self.transform_text_to_vector(self.x)
        clf = SVC(random_state=0, probability=True).fit(vectors, self.y)
        self.classifier = clf
        return self


class LSTMClassifier(Classifier):

    def train(self, x=None, y=None):
        pass

    def load_model(self):
        pass

    def predict(self, texts):
        pass


if __name__ == '__main__':
    from src.data.dataset import Tokenizer
    data = Sentences.read_train_data()
    DeepModel(word_vector=WordVector(), config=DeepModelConfig(), tokenizer=Tokenizer().tokenize).train(x=data["sentence"].values, y=data["label"].values).evaluate()
    # # print(score)
    # FastTextClassifier(self_train_model_path).evaluate()
    # print(FastTextClassifier().get_dirty_word_in_the_model(threshold=0.45))
