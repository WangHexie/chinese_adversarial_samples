import random
import numpy as np
from string import punctuation

from pypinyin import lazy_pinyin
from gensim.models import KeyedVectors

from src.config.configs import no_chinese_tokenizer_word_tf_idf_config, single_character_tf_idf_config
from src.data.dataset import Sentences
from src.predict.classifier import FastTextClassifier
from src.predict.word_vector import WordVector
from src.train.identify_importance_word import InspectFeatures

dirty_character_list = InspectFeatures(single_character_tf_idf_config).locate_top_dirty_character()
dirty_word_list = InspectFeatures(no_chinese_tokenizer_word_tf_idf_config).locate_top_dirty_character()


def insert_sting_middle(string, word, index):
    return string[:index] + word + string[index:]


class PaperRealize:

    def __init__(self, classifier, word_vector):
        self.classifier = classifier
        self.word_vector = word_vector

    @staticmethod
    def tokenize_text(text, method=0):
        if method == 0:
            return list(text)

    def __identify_important_word(self, text_list):
        # todo: add stop word list
        leave_one_text_list = ["".join(text_list)]

        for i in range(len(text_list)):
            text = text_list.copy()
            # act as out-of-vocabulary
            text[i] = '叇'
            leave_one_text_list.append("".join(text))

        probs = self.classifier.predict(leave_one_text_list)
        origin_prob = probs[0]
        leave_one_probs = probs[1:]
        return origin_prob - np.array(leave_one_probs)

    def __replace_text_and_predict(self, text_tokens, synonyms, index):
        sentences = []
        for word in synonyms:
            temp = text_tokens.copy()
            temp[index] = word
            sentences.append("".join(temp))
        return self.classifier.predict(sentences)

    @staticmethod
    def __choose_synonym_to_replace(scores, original_scores, score_threshold=0.4):
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
    def __replace_with_synonym(text_token, index_to_replace, synonyms, index_of_synonyms):
        if index_of_synonyms is None:
            return text_token
        text_token[index_to_replace] = synonyms[index_of_synonyms]
        return text_token

    def craft_one_adversarial_sample(self, text, top_num_of_word_to_modify=5):
        origin_score = self.classifier.predict([text])[0]

        # tokenize text
        tokenized_text = self.tokenize_text(text)

        # identify important word
        important_score = self.__identify_important_word(tokenized_text)

        important_word_index = np.argsort(important_score)[::-1]

        for i in range(min(len(tokenized_text), top_num_of_word_to_modify)):
            # find synonyms of important word
            try:
                synonyms = self.word_vector.most_similar(tokenized_text[important_word_index[i]], topn=5)
            except KeyError:
                continue
            # replace and predict
            scores = self.__replace_text_and_predict(tokenized_text, synonyms, i)

            synonym_index, has_succeeded = self.__choose_synonym_to_replace(scores, origin_score)
            tokenized_text = self.__replace_with_synonym(tokenized_text, i, synonyms, synonym_index)
            if has_succeeded:
                break

        return "".join(tokenized_text)


def replace_dirty_word(sentences):
    global dirty_character_list
    global dirty_word_list

    for k in dirty_word_list:
        sentences = sentences.replace(k, insert_sting_middle(k, random.choice(punctuation), 1))

    # for k in dirty_character_list:
    #     sentences = sentences.replace(k, pinyin(k, style=Style.FIRST_LETTER)[0][0])
    for k in dirty_character_list:
        sentences = sentences.replace(k, lazy_pinyin(k)[0])

    return sentences


if __name__ == '__main__':
    pr = PaperRealize(FastTextClassifier(), word_vector=WordVector())
    data = Sentences.read_insult_data()
    p = data["sentence"].map(lambda x: pr.craft_one_adversarial_sample(x))
    scores = pr.classifier.predict(p.values)
    print(p)
