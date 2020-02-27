import abc
import random

import numpy as np
from pypinyin import lazy_pinyin

from src.data.dataset import Sentences
from src.features.identify_importance_word import PrepareWords
from src.predict.word_vector import WordVector


def insert_sting_middle(string, word, index):
    return string[:index] + word + string[index:]


class RuleBased:
    @abc.abstractmethod
    def replace(self, sentences):
        pass


class ReplaceWithSynonyms(RuleBased):
    def __init__(self, word_vector: WordVector):
        self.word_vector = word_vector
        self.bad_words = PrepareWords.get_full_bad_words_and_character(2000)
        self.random_limit = 10
        self.use_limit = 20  # TODO: not enabled now

        self.special_word = "你"  # TODO: enable special word function

    def replace(self, sentences):

        for word in self.bad_words:
            try:
                similar_word = self.word_vector.find_synonyms_with_word_count_and_limitation(
                    word,
                    topn=self.random_limit)

                sentences = sentences.replace(word, random.choice(similar_word))

            except ValueError:
                continue
            except IndexError:
                continue

        return sentences


class DeleteDirtyWord(RuleBased):
    def __init__(self):
        self.full_words = PrepareWords.get_full_bad_words_and_character()

    def replace(self, sentences):
        dirty_word_list = self.full_words

        for k in dirty_word_list:
            sentences = sentences.replace(k, '')

        return sentences


class DeleteAll(RuleBased):
    def __init__(self):
        self.dirty_character_list = PrepareWords.get_dirty_character_list()
        self.dirty_word_list = PrepareWords.get_dirty_word_list()

    def replace(self, sentences):
        dirty_character_list = self.dirty_character_list
        dirty_word_list = self.dirty_word_list

        for k in dirty_word_list:
            sentences = sentences.replace(k, '')

        for k in dirty_character_list:
            sentences = sentences.replace(k, '')

        return sentences


class ReplaceWithPhonetic(RuleBased):
    def __init__(self):
        self.common_word = Sentences.read_common_words()
        self.full_word = PrepareWords.get_full_bad_words_and_character()

    def _find_sound_like_word(self, word_to_replace, random_limit):
        """
        need to be in common word list for readability
        :param word_to_replace:
        :param random_limit:
        :return:
        """
        common_word_list = self.common_word
        similar_word_list = common_word_list

        max_index = len(similar_word_list) - 1
        index = similar_word_list.index(word_to_replace)
        movement = random.randint(-random_limit, random_limit)
        if index + movement > max_index:
            movement = random.randint(index - random_limit, max_index) - index
        replace_word = similar_word_list[index + movement]
        same_pinyin_start = self._is_same_pinyin_start(word_to_replace, replace_word)
        if not same_pinyin_start:
            return self._find_sound_like_word(word_to_replace, random_limit)
        else:
            return similar_word_list[index + movement]

    @staticmethod
    def _is_same_pinyin_start(word, word_n):
        return lazy_pinyin(word)[0][0] == lazy_pinyin(word_n)[0][0]

    def replace(self, sentences):
        random_limit = 10

        dirty_word_list = self.full_word

        for word in dirty_word_list:
            word_to_replace = random.choice(word)
            try:

                similar_word = self._find_sound_like_word(word_to_replace, random_limit)

                sentences = sentences.replace(word, word.replace(word_to_replace, similar_word))

            except ValueError:
                continue
        return sentences


class RandomAppendGoodWords(RuleBased):
    def __init__(self, number_to_append=0.99):
        self.number_to_append = number_to_append
        self.good_word_and_character_list = PrepareWords.get_good_word_and_character_list()

    def replace(self, sentences):
        good_word_and_character = self.good_word_and_character_list
        count = 0
        if type(self.number_to_append) == float:
            number_to_append = int(self.number_to_append * len(sentences))
        else:
            number_to_append = self.number_to_append

        for gwc in good_word_and_character:
            if gwc in sentences:
                sentences = sentences + "" + gwc
                count += 1

            if number_to_append == count:
                break

        return sentences


class SimpleDeleteAndReplacement:
    # dirty_character_list = InspectFeatures(single_character_tf_idf_config).locate_top_dirty_character(70)
    # dirty_word_list = InspectFeatures.is_dirty_by_classifier(FastTextClassifier(self_train_model_path),
    #                                                          InspectFeatures(
    #                                                              no_chinese_tokenizer_word_tf_idf_config).locate_top_dirty_character(
    #                                                              1000),
    #                                                          0.1)
    # good_word_and_character_list = InspectFeatures(full_tokenizer_word_tf_idf_config,
    #                                                number_of_positive_data=-1).locate_top_not_dirty_character(-1)
    #
    # full_words = InspectFeatures(full_word_tf_idf_config, number_of_positive_data=-1).locate_top_dirty_character(1000)
    # shape_like_word = list(Sentences.read_similar_word())
    # common_word = Sentences.read_common_words()

    @staticmethod
    def delete_all_at_a_time(sentences):
        pass

    @staticmethod
    def delete_dirty_word(sentences):
        pass

    @staticmethod
    def replace_with_shape_like_or_sound_like_word(sentences):
        pass

    @staticmethod
    def replace_dirty_word(sentences):
        dirty_character_list = SimpleDeleteAndReplacement.dirty_character_list
        dirty_word_list = SimpleDeleteAndReplacement.dirty_word_list

        punctuation = ",\.\/\'~!@#$%()]["

        # for k in dirty_word_list:
        #     sentences = sentences.replace(k, insert_sting_middle(k, random.choice(punctuation), 1))

        for k in dirty_character_list:
            sentences = sentences.replace(k, lazy_pinyin(k)[0])

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
    def add_good_word(string):
        return "你这个人真好啊。" + string + "好棒，加油哦"

    @staticmethod
    def random_append_good_word(string, number_to_append=0.99):
        pass
