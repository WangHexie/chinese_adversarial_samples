import os
import random
from dataclasses import asdict

import numpy as np

from src.config.configs import SOTAAttackConfig, TFIDFConfig, full_word_tf_idf_config, self_train_model_path, \
    DeepModelConfig
from src.data.dataset import Sentences, Tokenizer
from src.data.measure_distance import DistanceCalculator
from src.manipulate.importance_based import ReplacementEnsemble
from src.manipulate.rule_based import DeleteDirtyWordFoundByTokenizer, \
    ListOfSynonyms, RandomAppendGoodWords
from src.models.classifier import FastTextClassifier, TFIDFClassifier, EmbeddingSVM, DeepModel
from src.predict.word_vector import WordVector


class EvaluateAttack:
    @staticmethod
    def choose_dataset(data_type=0):
        if data_type == 0:
            return Sentences.read_insult_data()

        if data_type == 1:
            data = Sentences.read_test_data()
            data = data[data["label"] == 1]
            print("test data length", len(data))
            return data

    @staticmethod
    def _evaluate_distance_and_score(scores, original_sentences, adversarial_sentences):
        success_status = np.array(scores) < 0.5
        success_number = success_status.sum()
        success_rate = success_number / len(scores)
        print("success rate:", success_rate)

        distances = DistanceCalculator()(list(original_sentences), list(adversarial_sentences))
        final_score = np.dot(distances['final_similarity_score'], success_status) / len(scores)
        print("final_score:", final_score)

        print("avg distance:", final_score / success_rate)

    @staticmethod
    def evaluate(attack_method: callable, classifier, dataset_type=0):
        data = EvaluateAttack.choose_dataset(dataset_type)

        original_sentences = data["sentence"].values.tolist()

        scores = classifier.predict(original_sentences)
        success_status = np.array(scores) > 0.5
        success_number = success_status.sum()
        print("original classifier score:", success_number / len(data))

        adversarial_sentences = data["sentence"].map(lambda x: attack_method(x)).values.tolist()
        scores = classifier.predict(adversarial_sentences)

        EvaluateAttack._evaluate_distance_and_score(scores, original_sentences, adversarial_sentences)
        print("----------- success in the success ------------------")
        success_status = np.array(success_status)
        scores = np.array(scores)
        EvaluateAttack._evaluate_distance_and_score(scores[success_status],
                                                    np.array(original_sentences)[success_status],
                                                    np.array(adversarial_sentences)[success_status])


def self_defined_function(manipulate_func: callable):
    data = Sentences.read_train_data()
    classifiers = {"fasttext provided": FastTextClassifier(),
                   "fasttext self-trained": FastTextClassifier(self_train_model_path),
                   "embedding svn": EmbeddingSVM(x=data["sentence"],
                                                  y=data["label"], word_vector=WordVector()).train(),
                   "tfidf ngram": TFIDFClassifier(tf_idf_config=asdict(TFIDFConfig(ngram_range=(1, 3), min_df=0.0005)),
                                                  x=data["sentence"],
                                                  y=data["label"]).train(),
                   "tfidf tokenizer": TFIDFClassifier(tf_idf_config=full_word_tf_idf_config,
                                                      x=data["sentence"],
                                                      y=data["label"]).train(),

                   }

    control_group = {"delete all dirty word": DeleteDirtyWordFoundByTokenizer().replace}

    for key, item in classifiers.items():
        print("-----------start {} evaluate---------------".format(key))
        EvaluateAttack.evaluate(manipulate_func, item, dataset_type=1)

    for name, func in control_group.items():
        for cls_name, cls in classifiers.items():
            print("-----------control group {}, {} evaluate---------------".format(name, cls_name))
            EvaluateAttack.evaluate(func, cls, dataset_type=1)

    # print("-----------append fastext evaluate---------------")
    # EvaluateAttack.evaluate(SimpleDeleteAndReplacement.random_append_good_word, FastTextClassifier(), dataset_type=1)
    # print("-----------append tfidf evaluate---------------")
    # EvaluateAttack.evaluate(SimpleDeleteAndReplacement.random_append_good_word, cls, dataset_type=1)


if __name__ == '__main__':
    # data = Sentences.read_train_data()

    # config = SOTAAttackConfig(num_of_synonyms=40,
    #                           threshold_of_stopping_attack=0.001, tokenize_method=0, word_use_limit=20,
    #                           text_modify_percentage=0.35)
    #
    # pr = ReplacementEnsemble([
    #     FastTextClassifier(),
    #     EmbeddingSVM(x=data["sentence"],
    #                  y=data["label"], word_vector=WordVector()).train(),
    #     # FastTextClassifier().train(),
    #     # TFIDFEmbeddingClassifier(word_vector=WordVector(tencent_embedding_path), tf_idf_config=full_word_tf_idf_config,
    #     # x=data["sentence"],
    #     # y=data["label"]).train(),
    #     # TFIDFClassifier(x=data["sentence"], y=data["label"]).train(),
    #     TFIDFClassifier(tf_idf_config=asdict(TFIDFConfig(ngram_range=(1, 3),
    #                                                      min_df=0.0005)), x=data["sentence"],
    #                     y=data["label"]).train()
    # ],
    #     word_vector=WordVector(),
    #     attack_config=config,
    #     replacement_classes=[ListOfSynonyms(word_vector=WordVector(), attack_config=config)]
    # )
    # config = SOTAAttackConfig(num_of_synonyms=40,
    #                           threshold_of_stopping_attack=0.001, tokenize_method=1, word_use_limit=20,
    #                           text_modify_percentage=0.50)
    #
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    #
    # pr = ReplacementEnsemble([
    #     DeepModel(word_vector=WordVector(), config=DeepModelConfig(), tokenizer=Tokenizer().tokenize).train(
    #         x=data["sentence"].values, y=data["label"].values)
    # ],
    #     word_vector=WordVector(),
    #     attack_config=config,
    #     replacement_classes=[ListOfSynonyms(word_vector=WordVector(), attack_config=config)]
    # )
    p = RandomAppendGoodWords(number_to_append=2.5)

    func = DeleteDirtyWordFoundByTokenizer()


    def repp(string):
        return p.replace(func.replace(string))
    self_defined_function(repp)
    # ListOfSynonyms(word_vector=WordVector(), attack_config=SOTAAttackConfig(num_of_synonyms=40,
    #                                                                         threshold_of_stopping_attack=0.001,
    #                                                                         tokenize_method=1,
    #                                                                         word_use_limit=20,
    #                                                                         text_modify_percentage=0.5))
