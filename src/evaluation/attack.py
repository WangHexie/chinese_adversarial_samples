import numpy as np

from src.data.dataset import Sentences
from src.manipulate.black_box import PaperRealize
from src.predict.classifier import FastTextClassifier
from src.predict.word_vector import WordVector


def evaluate_paper_attack():
    pr = PaperRealize(FastTextClassifier(), word_vector=WordVector())
    data = Sentences.read_insult_data()
    p = data["sentence"].map(lambda x: pr.craft_one_adversarial_sample(x))
    scores = pr.classifier.predict(p.values.tolist())
    success_number = (np.array(scores) < 0.5).sum()
    print("success rate:", success_number/len(data))


if __name__ == '__main__':
    evaluate_paper_attack()
