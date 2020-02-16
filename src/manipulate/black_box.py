import random
import numpy as np
from string import punctuation

from pypinyin import lazy_pinyin

from src.config.configs import no_chinese_tokenizer_word_tf_idf_config, single_character_tf_idf_config
from src.train.identify_importance_word import InspectFeatures

dirty_character_list = InspectFeatures(single_character_tf_idf_config).locate_top_dirty_character()
dirty_word_list = InspectFeatures(no_chinese_tokenizer_word_tf_idf_config).locate_top_dirty_character()


def insert_sting_middle(string, word, index):
    return string[:index] + word + string[index:]


class PaperRealize:

    def __init__(self, classifier):
        self.classifier = classifier

    @staticmethod
    def tokenize_text(text, method=0):
        if method == 0:
            return list(text)

    def identify_important_word(self, text_list):
        leave_one_text_list = [text_list]

        for i in range(len(text_list)):
            text = text_list.copy()
            text.pop(i)
            leave_one_text_list.append(''.join(text))

        probs = self.classifier.predict(leave_one_text_list)
        origin_prob = probs[0]
        leave_one_probs = probs[1:]
        return origin_prob - np.array(leave_one_probs)

    def craft_one_adversarial_sample(self, text):
        # tokenize text
        tokenized_text = self.tokenize_text(text)

        # identify important word
        important_score = self.identify_important_word(tokenized_text)

        # find synonyms of important word


        # replace and predict

        return


def replace_dirty_word(sentences):
    global dirty_character_list
    global dirty_word_list

    # dirty_replace_dictionary = {"马": "ma", "妈": "ma", "爸": "ba", "孙": "sun", "菊": "ju", "肛": "gang"}
    # for k, v in dirty_replace_dictionary.items():
    #     sentences = sentences.replace(k, v)

    for k in dirty_word_list:
        sentences = sentences.replace(k, insert_sting_middle(k, random.choice(punctuation), 1))

    # for k in dirty_character_list:
    #     sentences = sentences.replace(k, pinyin(k, style=Style.FIRST_LETTER)[0][0])
    for k in dirty_character_list:
        sentences = sentences.replace(k, lazy_pinyin(k)[0])

    return sentences


if __name__ == '__main__':
    print(1-np.array([1,2,3,4]))