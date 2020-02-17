import numpy as np

from src.config.configs import strong_attack_config
from src.data.dataset import Sentences
from src.data.measure_distance import DistanceCalculator
from src.manipulate.black_box import PaperRealize, replace_dirty_word, SelfDefined
from src.predict.classifier import FastTextClassifier
from src.predict.word_vector import WordVector


class EvaluateAttack:
    @staticmethod
    def evaluate(attack_method: callable, classifier):
        data = Sentences.read_insult_data()

        original_sentences = data["sentence"].values.tolist()
        adversarial_sentences = data["sentence"].map(lambda x: attack_method(x)).values.tolist()
        scores = classifier.predict(adversarial_sentences)

        success_status = np.array(scores) < 0.5
        success_number = success_status.sum()
        print("success rate:", success_number / len(data))

        distances = DistanceCalculator()(original_sentences, adversarial_sentences)
        print("final_score:", np.dot(distances['final_similarity_score'], success_status)/len(data))


if __name__ == '__main__':
    pr = SelfDefined(FastTextClassifier(), word_vector=WordVector(), attack_config=strong_attack_config)
    EvaluateAttack.evaluate(pr.craft_one_adversarial_sample, pr.classifier)
