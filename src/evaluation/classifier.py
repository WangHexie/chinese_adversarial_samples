import numpy as np
from sklearn.metrics import accuracy_score

from src.data.dataset import Sentences
from src.manipulate.black_box import replace_dirty_word
from src.predict.classifier import FastTextClassifier


def evaluate_classifier():
    datas = Sentences.read_full_data()
    preds = FastTextClassifier().load_model().predict(datas["sentence"].values.tolist())
    score = accuracy_score(datas["label"].values, np.array(preds).round())
    print(score)

    datas["sentence"] = datas["sentence"].map(lambda x:replace_dirty_word(x))
    preds = FastTextClassifier().load_model().predict(datas["sentence"].values.tolist())
    score = accuracy_score(datas["label"].values, np.array(preds).round())
    print(score)


if __name__ == '__main__':
    evaluate_classifier()
