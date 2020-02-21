import math
import random

import numpy as np
from pypinyin import lazy_pinyin

from src.config.configs import no_chinese_tokenizer_word_tf_idf_config, single_character_tf_idf_config, \
    SOTAAttackConfig, strong_attack_config, self_train_model_path, full_tokenizer_word_tf_idf_config
from src.data.dataset import Sentences, Tokenizer
from src.models.classifier import FastTextClassifier, TFIDFClassifier
from src.predict.word_vector import WordVector
from src.train.identify_importance_word import InspectFeatures


def insert_sting_middle(string, word, index):
    return string[:index] + word + string[index:]


def tokenizer_selector(method) -> callable:
    if method == 0:
        return list

    if method == 1:
        return Tokenizer().tokenize


class ImportanceBased:

    def __init__(self, classifiers, word_vector, attack_config: SOTAAttackConfig):
        self.classifiers = classifiers
        self.word_vector = word_vector
        self.attack_config = attack_config
        self.tokenizer = tokenizer_selector(method=self.attack_config.tokenize_method)
        self.stop_word_and_structure_word_list = ["你"]

    def tokenize_text(self, text):
        return self.tokenizer(text)

    def _identify_important_word(self, text_list, word_to_replace='叇'):
        # todo: add stop word list
        leave_one_text_list = ["".join(text_list)]

        for i in range(len(text_list)):
            text = text_list.copy()
            # act as out-of-vocabulary
            text[i] = word_to_replace
            leave_one_text_list.append("".join(text))

        probs = self.predict_use_classifiers(leave_one_text_list)

        origin_prob = probs[0]
        leave_one_probs = probs[1:]
        importance_score = origin_prob - np.array(leave_one_probs)
        for i in range(len(importance_score)):
            if text_list[i] in self.stop_word_and_structure_word_list:
                importance_score[i] = 0
        return importance_score

    def predict_use_classifiers(self, text_list):
        return np.mean([classifier.predict(text_list) for classifier in self.classifiers], axis=0)

    def _temporal_head_score(self, text_list):
        top_n = [''.join(text_list[:i + 1]) for i in range(len(text_list))]
        probs = self.predict_use_classifiers(top_n)

    def _replace_text_and_predict(self, text_tokens, synonyms, index):
        sentences = []
        for word in synonyms:
            temp = text_tokens.copy()
            temp[index] = word
            sentences.append("".join(temp))
        return self.predict_use_classifiers(sentences)

    @staticmethod
    def _choose_synonym_to_replace(scores, original_scores, score_threshold=0.4):
        """
        Todo: choose a real one
        :param scores:
        :param original_scores:
        :param score_threshold:
        :return: index_of synonym , control character
        """
        if min(scores) > original_scores:
            return None, None

        if min(scores) < score_threshold:
            return np.argmin(scores), True
        return np.argmin(scores), False

    @staticmethod
    def _replace_with_synonym(text_token, index_to_replace, synonyms, index_of_synonyms):
        if index_of_synonyms is None:
            return text_token
        text_token[index_to_replace] = synonyms[index_of_synonyms]
        return text_token

    def craft_one_adversarial_sample(self, text):
        # identify, replace_and
        origin_score = self.predict_use_classifiers([text])[0]

        # tokenize text
        tokenized_text = self.tokenize_text(text)

        # identify important word
        important_score = self._identify_important_word(tokenized_text)

        important_word_index = np.argsort(important_score)[::-1]

        for i in range(math.floor(self.attack_config.text_modify_percentage * len(tokenized_text))):
            # find synonyms of important word

            index_of_word_to_replace = important_word_index[i]
            try:
                synonyms = self.word_vector.most_similar(tokenized_text[index_of_word_to_replace],
                                                         topn=self.attack_config.num_of_synonyms)
            except KeyError:
                continue
            # replace and predict
            scores = self._replace_text_and_predict(tokenized_text, synonyms, index_of_word_to_replace)

            synonym_index, has_succeeded = self._choose_synonym_to_replace(scores, origin_score,
                                                                           self.attack_config.threshold_of_stopping_attack)
            tokenized_text = self._replace_with_synonym(tokenized_text, index_of_word_to_replace, synonyms,
                                                        synonym_index)
            if has_succeeded:
                break

        return "".join(tokenized_text)


class RemoveImportantWord(ImportanceBased):
    def craft_one_adversarial_sample(self, text):
        origin_score = self.predict_use_classifiers([text])[0]

        word_to_replace = ['']

        # tokenize text
        tokenized_text = self.tokenize_text(text)

        # identify important word
        important_score = self._identify_important_word(tokenized_text, word_to_replace='')
        important_word_index = np.argsort(important_score)[::-1]

        for i in range(min(len(tokenized_text), self.attack_config.top_num_of_word_to_modify)):
            # find synonyms of important word

            index_of_word_to_replace = important_word_index[i]

            synonyms = word_to_replace

            # replace and predict
            scores = self._replace_text_and_predict(tokenized_text, synonyms, index_of_word_to_replace)

            synonym_index, has_succeeded = self._choose_synonym_to_replace(scores,
                                                                           origin_score,
                                                                           self.attack_config.threshold_of_stopping_attack)
            tokenized_text = self._replace_with_synonym(tokenized_text, index_of_word_to_replace, synonyms,
                                                        synonym_index)
            if has_succeeded:
                break

        return "".join(tokenized_text)


class SimpleDeleteAndReplacement:
    dirty_character_list = InspectFeatures(single_character_tf_idf_config).locate_top_dirty_character(70)
    dirty_word_list = InspectFeatures.is_dirty_by_classifier(FastTextClassifier(self_train_model_path),
                                                             InspectFeatures(
                                                                 no_chinese_tokenizer_word_tf_idf_config).locate_top_dirty_character(
                                                                 1000),
                                                             0.1)
    good_word_and_character_list = InspectFeatures(full_tokenizer_word_tf_idf_config,
                                                   number_of_positive_data=-1).locate_top_not_dirty_character()

    @staticmethod
    def delete_dirty_word(sentences):
        dirty_character_list = SimpleDeleteAndReplacement.dirty_character_list
        dirty_word_list = SimpleDeleteAndReplacement.dirty_word_list

        # for k in dirty_word_list:
        #     sentences = sentences.replace(k, insert_sting_middle(k, random.choice(punctuation), 1))
        #
        # for k in dirty_character_list:
        #     sentences = sentences.replace(k, lazy_pinyin(k)[0])

        for k in dirty_word_list:
            sentences = sentences.replace(k, '')

        for k in dirty_character_list:
            sentences = sentences.replace(k, '')

        return sentences

    @staticmethod
    def replace_dirty_word(sentences):
        dirty_character_list = SimpleDeleteAndReplacement.dirty_character_list
        dirty_word_list = SimpleDeleteAndReplacement.dirty_word_list

        punctuation = ",\.\/\'~!@#$%()]["

        for k in dirty_word_list:
            sentences = sentences.replace(k, insert_sting_middle(k, random.choice(punctuation), 1))

        for k in dirty_character_list:
            sentences = sentences.replace(k, SimpleDeleteAndReplacement.random_upper_case(lazy_pinyin(k)[0]))

        return sentences

    @staticmethod
    def random_upper_case(string):
        upper_ratio = 0.5
        str_len = len(string)

        num = int(upper_ratio * str_len)
        upper_index = np.random.choice(list(range(len(string))), num)
        for i in upper_index:
            string = string[:i] + string[i].upper() + string[i + 1:]

        return string

    @staticmethod
    def random_append_good_word(string, number_to_append=8):
        good_word_and_character = SimpleDeleteAndReplacement.good_word_and_character_list
        count = 0
        for gwc in good_word_and_character:
            if gwc in string:
                string = string + "。" + gwc
                count += 1

            if number_to_append == count:
                break

        return string


if __name__ == '__main__':
    # print(random_upper_case("what are you takkk"))
    data = Sentences.read_train_data()
    pr = RemoveImportantWord([FastTextClassifier(self_train_model_path),
                              TFIDFClassifier(x=data["sentence"], y=data["label"]).train(),
                              TFIDFClassifier(tf_idf_config=no_chinese_tokenizer_word_tf_idf_config, x=data["sentence"],
                                              y=data["label"]).train()],
                             word_vector=WordVector(),
                             attack_config=strong_attack_config)
    data = Sentences.read_insult_data()

    print(SimpleDeleteAndReplacement.dirty_character_list)
    print(SimpleDeleteAndReplacement.dirty_word_list)
    p = data["sentence"].map(lambda x: pr.craft_one_adversarial_sample(x))
    # scores = pr.classifier.predict(p.values.tolist())

    print(len(InspectFeatures.is_dirty_by_classifier(pr.classifiers[0], SimpleDeleteAndReplacement.dirty_character_list,
                                                     0.4)))
    print(
        len(InspectFeatures.is_dirty_by_classifier(pr.classifiers[0], SimpleDeleteAndReplacement.dirty_word_list, 0.4)))
    print(p.values.tolist()[:30])
    print(data["sentence"].values.tolist()[:30])
