import abc

import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config.configs import single_character_tf_idf_config, full_word_tf_idf_config, \
    no_chinese_tokenizer_word_tf_idf_config, full_tokenizer_word_tf_idf_config, tencent_embedding_path
from src.data.dataset import Sentences
from src.models.classifier import FastTextClassifier, DeepModel, TFIDFEmbeddingClassifier
from src.predict.word_vector import WordVector


class InspectFeatures:
    def __init__(self, tf_idf_config=single_character_tf_idf_config, number_of_positive_data=2371):
        self.tf_idf_config = tf_idf_config
        self.number_of_positive_data = number_of_positive_data

    @staticmethod
    def tf_idf_features(text, tf_idf_config):
        vectorizer = TfidfVectorizer(input='content', encoding='utf-8', decode_error='ignore', lowercase=False,
                                     **tf_idf_config)
        text_features = vectorizer.fit_transform(text)
        feature_names = vectorizer.get_feature_names()
        print("identified feature length:", len(feature_names))
        return text_features, feature_names

    @staticmethod
    def find_important_features_by_using_linear_model(features, labels):
        clf = LogisticRegression(random_state=0).fit(features, labels)
        print("classifier score:", clf.score(features, labels))
        return clf.coef_

    @staticmethod
    def is_dirty_by_classifier(classifier, dirty_list, threshold=0.4):
        scores = classifier.predict(dirty_list)
        is_dirty = np.array(scores) > threshold

        not_dirty = [dirty_list[i] for i in range(len(scores)) if not is_dirty[i]]

        print("not dirty:", len(not_dirty), not_dirty)
        return [dirty_list[i] for i in range(len(scores)) if is_dirty[i]]

    def _locate_top_character(self, number_of_dirty=70, choose_dirty=True):
        """

        :param number_of_dirty: -1:all
        :param choose_dirty:
        :return:
        """
        data = Sentences.read_full_data(num_of_positive=self.number_of_positive_data)
        text_features, feature_names = InspectFeatures.tf_idf_features(data["sentence"], self.tf_idf_config)
        coef = InspectFeatures.find_important_features_by_using_linear_model(text_features, data["label"])[0]
        if not choose_dirty:
            coef = -coef
        dirty_word_index = np.argsort(coef)[::-1]
        return [feature_names[dirty_word_index[i]] for i in range(len(coef)) if coef[dirty_word_index[i]] > 0][
               :number_of_dirty]

    def locate_top_not_dirty_character(self, number=1000):
        """

        :param number: -1:all
        :return:
        """
        return self._locate_top_character(number_of_dirty=number, choose_dirty=False)

    def locate_top_dirty_character(self, number_of_dirty=70):
        """

        :param number_of_dirty: -1:all
        :return:
        """
        return self._locate_top_character(number_of_dirty, choose_dirty=True)


class FindDirtyWordInEmbedding():
    def __init__(self, word_vector: WordVector, config):
        self.word_vector = word_vector
        self.full_word = list(word_vector.vector.vocab.keys())
        x, y = Sentences.get_word_pos_neg_data()
        self.classifier = TFIDFEmbeddingClassifier(word_vector=word_vector, tf_idf_config=config, x=x, y=y)

    def get_all_dirty_word_in_embedding(self, threshold=0.92):
        classifier = self.classifier.train()
        scores = np.array(classifier.predict(self.full_word))
        is_dirty_word = scores > threshold
        dirty_word = np.array(self.full_word)[is_dirty_word]
        dirty_word_score = scores[is_dirty_word]

        score_index = dirty_word_score.argsort()[::-1]
        return dirty_word[score_index]


class PrepareWords:
    @staticmethod
    def get_dirty_character_list(number_of_characters=70):
        return InspectFeatures(single_character_tf_idf_config).locate_top_dirty_character(number_of_characters)

    @staticmethod
    def get_full_dirty_word_list_by_ngram(number_of_characters=1000):
        return InspectFeatures(full_tokenizer_word_tf_idf_config).locate_top_dirty_character(number_of_characters)

    @staticmethod
    def get_dirty_word_list(number_of_characters=1000, classifier_threshold=0.1):
        return InspectFeatures.is_dirty_by_classifier(FastTextClassifier(),
                                                      InspectFeatures(
                                                          no_chinese_tokenizer_word_tf_idf_config).locate_top_dirty_character(
                                                          number_of_characters),
                                                      classifier_threshold)

    @staticmethod
    def get_good_word_and_character_list():
        return InspectFeatures(full_tokenizer_word_tf_idf_config,
                               number_of_positive_data=-1).locate_top_not_dirty_character(-1)

    @staticmethod
    def get_full_bad_words_and_character(number_of_characters=1000):
        return InspectFeatures(full_word_tf_idf_config, number_of_positive_data=-1).locate_top_dirty_character(
            number_of_characters)

    @staticmethod
    def get_dirty_word_in_the_classifier(threshold=0.45):
        return FastTextClassifier().get_dirty_word_in_the_model(threshold=threshold)

    @staticmethod
    def delete_stop_word_in_special_word(full_word, stop_word) -> list:
        result = []
        for bad_word in full_word:
            contain = 0
            for stop_word in stop_word:
                if stop_word in bad_word:
                    contain = 1
                    break
            if not contain:
                result.append(bad_word)

        return result

    @staticmethod
    def get_full_dirty_without_stop_word():
        full_word = PrepareWords.get_dirty_word_in_the_classifier() + PrepareWords.get_full_bad_words_and_character() + PrepareWords.get_full_dirty_word_list_by_ngram(
            1000)
        return PrepareWords.delete_stop_word_in_special_word(full_word, Sentences.read_stop_words())


class ImportanceJudgement:
    @abc.abstractmethod
    def identify_important_word(self, word_list):
        pass


class FGSM(ImportanceJudgement):
    def identify_important_word(self, word_list):
        word_indexes = list(map(lambda x: DeepModel.map_func(x, self.dictionary), word_list))
        train_data_i = tf.keras.preprocessing.sequence.pad_sequences([word_indexes], maxlen=self.input_length,
                                                                     dtype='int32',
                                                                     padding='post', truncating='pre', value=0)
        tensor_t = tf.convert_to_tensor(train_data_i)

        embeddings = self.embedding_layer(tensor_t)
        loss_object = tf.keras.losses.mean_absolute_error

        with tf.GradientTape() as tape:
            tape.watch(embeddings)
            prediction = self.middle_layer(embeddings)
            loss = loss_object([[1]], prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, embeddings)
        sqr = tf.keras.backend.square(gradient)
        sqr_sum = tf.math.reduce_sum(sqr, 2)
        return sqr_sum.numpy()[0][:len(word_list)].argsort()[::-1]

    def __init__(self, embedding_layer, middle_layer, dictionary, input_length):
        self.embedding_layer = embedding_layer
        self.middle_layer = middle_layer
        self.dictionary = dictionary
        self.input_length = input_length


if __name__ == '__main__':
    # print(InspectFeatures(full_word_tf_idf_config, number_of_positive_data=-1).locate_top_dirty_character(800))
    # print(len(PrepareWords.get_dirty_word_in_the_classifier()))
    print(FindDirtyWordInEmbedding(WordVector(tencent_embedding_path),
                                   config=full_word_tf_idf_config).get_all_dirty_word_in_embedding())
