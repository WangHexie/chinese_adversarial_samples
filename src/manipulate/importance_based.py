import math
from typing import List

import numpy as np

from src.config.configs import SOTAAttackConfig, full_word_tf_idf_config, tencent_embedding_path
from src.data.dataset import Sentences, Tokenizer
from src.manipulate.rule_based import ReplaceWithPhonetic, CreateListOfDeformation, InsertPunctuation, PhoneticList
from src.models.classifier import TFIDFClassifier
from src.predict.word_vector import WordVector


def tokenizer_selector(method) -> callable:
    if method == 0:
        return list

    if method == 1:
        return Tokenizer().tokenize


class ImportanceBased:

    def __init__(self, classifiers, word_vector: WordVector, attack_config: SOTAAttackConfig):
        self.classifiers = classifiers
        self.word_vector = word_vector
        self.attack_config = attack_config
        self.tokenizer = tokenizer_selector(method=self.attack_config.tokenize_method)
        self.stop_word_and_structure_word_list = ["你"]

    def tokenize_text(self, text):
        return self.tokenizer(text)

    def _identify_important_word(self, text_list, word_to_replace='叇'):
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
                importance_score[i] = -1
        return importance_score

    def predict_use_classifiers(self, text_list):
        return np.mean([classifier.predict(text_list) for classifier in self.classifiers], axis=0)

    def _temporal_head_score(self, text_list):
        # TODO: importance score not finished
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

    def _replace_with_synonym(self, text_token, index_to_replace, synonyms, index_of_synonyms):
        if index_of_synonyms is None:
            return text_token
        text_token[index_to_replace] = synonyms[index_of_synonyms]
        self.word_vector.add_word_use(synonyms[index_of_synonyms], self.attack_config.word_use_limit)
        return text_token

    def _find_synonyms_or_others_words(self, word):
        synonyms = self.word_vector.find_synonyms_with_word_count_and_limitation(
            word, topn=self.attack_config.num_of_synonyms)
        if len(synonyms) == 0:
            synonyms = self._modify_single_word_in_sentences(word)

        return synonyms

    def _modify_single_word_in_sentences(self, word):
        if len(word) == 1:
            return []

        result = []

        length = len(word)
        synonyms_num = int(self.attack_config.num_of_synonyms / length)
        for i in range(length):
            synonyms = self.word_vector.find_synonyms_with_word_count_and_limitation(word[i], topn=synonyms_num)
            if len(synonyms) == 0:
                continue
            for s in synonyms:
                result.append(word[:i] + s + word[i + 1:])

        return result

    def craft_one_adversarial_sample(self, text):
        # identify, replace_and
        origin_score = self.predict_use_classifiers([text])[0]

        # tokenize text
        tokenized_text = self.tokenize_text(text)

        # identify important word
        important_score = self._identify_important_word(tokenized_text, self.attack_config.word_to_replace)

        # only modify real important word

        important_word_index = np.argsort(important_score)[::-1]
        important_word_index = important_word_index[important_score[important_word_index] > 0]

        # np.array(tokenized_text)[important_word_index]

        for i in range(min(len(important_word_index),
                           math.floor(self.attack_config.text_modify_percentage * len(tokenized_text)))):
            # find synonyms of important word

            index_of_word_to_replace = important_word_index[i]

            synonyms = self._find_synonyms_or_others_words(tokenized_text[index_of_word_to_replace])

            if len(synonyms) == 0:
                # TODO: modification single word
                continue
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


class ReplacementEnsemble(ImportanceBased):
    def __init__(self,
                 classifiers,
                 word_vector: WordVector,
                 attack_config: SOTAAttackConfig,
                 replacement_classes: List[CreateListOfDeformation]):
        super().__init__(classifiers, word_vector, attack_config)
        self.replacement_classes = replacement_classes

    def _replace_with_synonym(self, text_token, index_to_replace, synonyms, index_of_synonyms):
        if index_of_synonyms is None:
            return text_token
        text_token[index_to_replace] = synonyms[index_of_synonyms]
        for r in self.replacement_classes:
            r.add_word_use(synonyms[index_of_synonyms], self.attack_config.word_use_limit)
        return text_token

    def _find_synonyms_or_others_words(self, word):
        result = []
        for r in self.replacement_classes:
            result += r.create(word)
        return result


class TFIDFBasedReplaceWithHomophone(ImportanceBased):
    def __init__(self, classifiers, word_vector: WordVector, attack_config: SOTAAttackConfig):
        super().__init__(classifiers, word_vector, attack_config)
        self.replacement = ReplaceWithPhonetic()

    def _find_synonyms_or_others_words(self, word):
        return [self.replacement.replace(word)]


class RemoveImportantWord(ImportanceBased):
    """
    set word_to_replace='' in config !!!!!!!!!!
    """

    def _find_synonyms_or_others_words(self, word):
        return ['']


if __name__ == '__main__':
    # print(random_upper_case("what are you takkk"))
    data = Sentences.read_full_data()
    pr = ReplacementEnsemble([
        # FastTextClassifier(self_train_model_path),
        # TFIDFClassifier(x=data["sentence"], y=data["label"]).train(),
        TFIDFClassifier(tf_idf_config=full_word_tf_idf_config, x=data["sentence"],
                        y=data["label"]).train()
    ],
        word_vector=WordVector(tencent_embedding_path),
        attack_config=SOTAAttackConfig(num_of_synonyms=40,
                                       threshold_of_stopping_attack=0.08, tokenize_method=1),
        replacement_classes=[InsertPunctuation(), PhoneticList()]
    )
    data = Sentences.read_insult_data().iloc[:500]

    # print(SimpleDeleteAndReplacement.dirty_character_list)
    # print(SimpleDeleteAndReplacement.dirty_word_list)
    # r = ReplaceWithSynonyms(word_vector=WordVector())
    p = data["sentence"].map(lambda x: pr.craft_one_adversarial_sample(x))
    # scores = pr.classifier.predict(p.values.tolist())

    # print(len(InspectFeatures.is_dirty_by_classifier(pr.classifiers[0], SimpleDeleteAndReplacement.dirty_character_list,
    #                                                  0.4)))
    # print(
    #     len(InspectFeatures.is_dirty_by_classifier(pr.classifiers[0], SimpleDeleteAndReplacement.dirty_word_list, 0.4)))
    print(p.values.tolist()[:30])
    print(data["sentence"].values.tolist()[:30])
