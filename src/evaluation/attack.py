import os

import numpy as np

from src.config.configs import strong_attack_config, self_train_model_path, tencent_embedding_path
from src.data.dataset import Sentences
from src.data.basic_functions import root_dir
from src.data.measure_distance import DistanceCalculator
from src.manipulate.black_box import PaperRealize, replace_dirty_word, SelfDefined
from src.models.classifier import FastTextClassifier
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
    def evaluate(attack_method: callable, classifier, dataset_type=0):
        data = EvaluateAttack.choose_dataset(dataset_type)

        original_sentences = data["sentence"].values.tolist()

        scores = classifier.predict(original_sentences)
        success_status = np.array(scores) > 0.5
        success_number = success_status.sum()
        print("original classifier score:", success_number / len(data))

        adversarial_sentences = data["sentence"].map(lambda x: attack_method(x)).values.tolist()
        scores = classifier.predict(adversarial_sentences)

        success_status = np.array(scores) < 0.5
        success_number = success_status.sum()
        print("success rate:", success_number / len(data))

        distances = DistanceCalculator()(original_sentences, adversarial_sentences)
        print("final_score:", np.dot(distances['final_similarity_score'], success_status)/len(data))


if __name__ == '__main__':
    pr = PaperRealize(FastTextClassifier(self_train_model_path),
                      word_vector=WordVector(tencent_embedding_path),
                      attack_config=strong_attack_config)
    EvaluateAttack.evaluate(pr.craft_one_adversarial_sample, pr.classifier, dataset_type=1)
    EvaluateAttack.evaluate(replace_dirty_word, pr.classifier, dataset_type=1)
