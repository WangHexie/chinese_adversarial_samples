import math
import os
from dataclasses import asdict
from typing import List

import jieba
import numpy as np

from src.config.configs import SOTAAttackConfig, full_word_tf_idf_config, tencent_embedding_path, TFIDFConfig, \
    DeepModelConfig
from src.data.dataset import Sentences, Tokenizer, Jieba
from src.features.identify_importance_word import ImportanceJudgement, FGSM
from src.manipulate.rule_based import ReplaceWithPhonetic, CreateListOfDeformation, InsertPunctuation, PhoneticList, \
    ListOfSynonyms
from src.models.classifier import TFIDFClassifier, FastTextClassifier, DeepModel
from src.models.deep_model import SimpleCnn, SimpleRNN
from src.predict.word_vector import WordVector


def tokenizer_selector(method) -> callable:
    if method == 0:
        return list

    if method == 1:
        return Tokenizer().tokenize

    if method == 2:
        return Jieba().tokenize()


class ImportanceBased:

    def __init__(self, classifiers, word_vector: WordVector, attack_config: SOTAAttackConfig):
        self.classifiers = classifiers
        self.word_vector = word_vector
        self.attack_config = attack_config
        self.tokenizer = tokenizer_selector(method=self.attack_config.tokenize_method)
        self.stop_word_and_structure_word_list = ["你", "我"] + Sentences.read_stop_words()

    def tokenize_text(self, text):
        return self.tokenizer(text)

    def _identify_important_word(self, text_list, word_to_replace='叇'):
        importance_score = np.mean([self._delete_or_replace_score(text_list, word_to_replace='叇'),
                                    self._temporal_head_score(text_list),
                                    self._temporal_tail_score(text_list)],  axis=0)

        for i in range(len(importance_score)):
            for word in self.stop_word_and_structure_word_list:
                if word in text_list[i]:
                    importance_score[i] = -1

        important_word_index = np.argsort(importance_score)[::-1]
        important_word_index = important_word_index[importance_score[important_word_index] > 0]
        return important_word_index

    def predict_use_classifiers(self, text_list):
        return np.mean([classifier.predict(text_list) for classifier in self.classifiers], axis=0)

    def _delete_or_replace_score(self, text_list, word_to_replace='叇'):
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
        return list(importance_score)

    def _temporal_head_score(self, text_list):
        top_n = [''.join(text_list[:i + 1]) for i in range(len(text_list))]
        probs = self.predict_use_classifiers(top_n)

        rear_n = list(probs)
        rear_n.insert(0, 0)
        rear_n.pop(-1)

        return list(np.array(probs) - np.array(rear_n))

    def _temporal_tail_score(self, text_list):
        top_n = [''.join(text_list[i:-1]) for i in range(len(text_list))]
        probs = self.predict_use_classifiers(top_n)

        rear_n = list(probs)
        rear_n.pop(0)
        rear_n.insert(-1, 0)

        return list(np.array(probs) - np.array(rear_n))


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
        important_word_index = self._identify_important_word(tokenized_text, self.attack_config.word_to_replace)

        for i in range(min(len(important_word_index),
                           math.floor(self.attack_config.text_modify_percentage * len(tokenized_text)))):
            # find synonyms of important word

            index_of_word_to_replace = important_word_index[i]

            synonyms = self._find_synonyms_or_others_words(tokenized_text[index_of_word_to_replace])

            if len(synonyms) == 0:
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
                 replacement_classes: List[CreateListOfDeformation],
                 classifier_coefficient: List = None):
        super().__init__(classifiers, word_vector, attack_config)
        self.replacement_classes = replacement_classes
        self.classifier_coefficient = classifier_coefficient

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

    def predict_use_classifiers(self, text_list):
        if self.classifier_coefficient is None:
            return np.mean([classifier.predict(text_list) for classifier in self.classifiers], axis=0)
        else:
            result = []
            for i in range(len(self.classifiers)):
                scores = (np.array(self.classifiers[i].predict(text_list)) - 0.5) * self.classifier_coefficient[i]
                result.append(scores)

            return np.mean(result, axis=0) + 0.5


class DeepImportanceScore(ReplacementEnsemble):
    def __init__(self, classifiers, word_vector: WordVector, attack_config: SOTAAttackConfig,
                 replacement_classes: List[CreateListOfDeformation], importance_judgement:ImportanceJudgement):
        super().__init__(classifiers, word_vector, attack_config, replacement_classes)
        self.importance_judgement = importance_judgement

    def _identify_important_word(self, text_list, word_to_replace='叇'):
        return self.importance_judgement.identify_important_word(text_list)
    
    
class ChooseToAppend(ReplacementEnsemble):
    def __init__(self, classifiers, word_vector: WordVector, attack_config: SOTAAttackConfig,
                 replacement_classes: List[CreateListOfDeformation]):

        super().__init__(classifiers, word_vector, attack_config, replacement_classes)
        self.good_word = Sentences.read_positive_word()
        self.word_to_choose = 30

    def craft_one_adversarial_sample(self, text):
        # identify, replace_and
        origin_score = self.predict_use_classifiers([text])[0]

        # tokenize text
        tokenized_text = self.tokenize_text(text)

        # identify important word

        for i in range(min(len(tokenized_text),
                           math.floor(self.attack_config.text_modify_percentage * len(tokenized_text)))):
            # find synonyms of important word

            words = np.random.choice(self.good_word, self.word_to_choose)

            sentences = [text+" "+ word for word in words]
            scores = origin_score - np.array(self.predict_use_classifiers(sentences))
            if max(scores) < 0:
                continue
            index_to_choose = scores.argmax()
            text = sentences[index_to_choose]

        return text


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
    # data = Sentences.read_full_data()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    #
    # data = Sentences.read_train_data()
    # dm = DeepModel(word_vector=WordVector(), config=DeepModelConfig(), tokenizer=Tokenizer().tokenize,
    #                model_creator=SimpleCnn).train(x=data["sentence"].values, y=data["label"].values)
    # pr = DeepImportanceScore([
    #     FastTextClassifier(),
    #     # TFIDFClassifier(x=data["sentence"], y=data["label"]).train(),
    #     # TFIDFClassifier(tf_idf_config=full_word_tf_idf_config, x=data["sentence"],
    #     #                 y=data["label"]).train()
    # ],
    #     word_vector=WordVector(tencent_embedding_path),
    #     attack_config=SOTAAttackConfig(num_of_synonyms=40,
    #                                    threshold_of_stopping_attack=0.08, tokenize_method=1),
    #     replacement_classes=[InsertPunctuation(), PhoneticList()],
    #     importance_judgement=FGSM(*dm.get_embedding_and_middle_layers(), dm.get_dictionary(), dm.config.input_length)
    # )
    data = Sentences.read_train_data()

    config = SOTAAttackConfig(num_of_synonyms=40,
                              threshold_of_stopping_attack=0.001, tokenize_method=1, word_use_limit=20,
                              text_modify_percentage=0.50)

    pr = ReplacementEnsemble([
        DeepModel(word_vector=WordVector(), config=DeepModelConfig(), tokenizer=list, model_creator=SimpleRNN).train(
            x=data["sentence"].values, y=data["label"].values)
    ],
        word_vector=WordVector(),
        attack_config=config,
        replacement_classes=[ListOfSynonyms(word_vector=WordVector(), attack_config=config)]
    )

    p = data["sentence"].map(lambda x: pr.craft_one_adversarial_sample(x))

    print(p.values.tolist()[:30])
    print(data["sentence"].values.tolist()[:30])
