import abc
import os

import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.config.configs import tencent_embedding_path, self_train_test_data_path, single_character_tf_idf_config, \
    strong_attack_config, self_train_model_path, self_train_train_data_path
from src.data.basic_functions import root_dir
from src.data.dataset import Sentences
from src.predict.word_vector import WordVector
import numpy as np


class Classifier:
    @abc.abstractmethod
    def load_model(self):
        print("error")
        return self

    @abc.abstractmethod
    def train(self):
        print("error")
        return self

    @abc.abstractmethod
    def predict(self, texts):
        print("error")
        return

    def evaluate(self):
        from src.manipulate.black_box import ImportanceBased, SimpleDeleteAndReplacement

        datas = Sentences.read_test_data()
        pr = ImportanceBased([self], word_vector=WordVector(), attack_config=strong_attack_config)

        preds = pr.classifiers[0].predict(datas["sentence"].values.tolist())
        score = accuracy_score(datas["label"].values, np.array(preds).round())
        print("untokenized score:", score)

        preds = pr.classifiers[0].predict(
            datas["sentence"].map(lambda x: ' '.join(pr.tokenize_text(x))).values.tolist())
        score = accuracy_score(datas["label"].values, np.array(preds).round())
        print("tokenize score:", score)

        datas["sentence"] = datas["sentence"].map(lambda x: SimpleDeleteAndReplacement.delete_all_at_a_time(x))
        preds = FastTextClassifier().load_model().predict(datas["sentence"].values.tolist())
        score = accuracy_score(datas["label"].values, np.array(preds).round())
        print("remove dirty word:", score)


class FastTextClassifier(Classifier):

    def __init__(self, model_path=os.path.join(root_dir(), "models", "mini.ftz")):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        model = fasttext.load_model(self.model_path)
        self.model = model
        return self

    def train(self):
        try:
            self.model = fasttext.train_supervised(self_train_train_data_path,
                                                   dim=200,
                                                   pretrainedVectors=tencent_embedding_path,
                                                   autotuneValidationFile=self_train_test_data_path,
                                                   autotuneDuration=1200)
        except Exception:
            Sentences().save_train_data()
            self.model = fasttext.train_supervised(self_train_train_data_path,
                                                   dim=200,
                                                   pretrainedVectors=tencent_embedding_path,
                                                   autotuneValidationFile=self_train_test_data_path,
                                                   autotuneDuration=3000)

        self.model.save_model(self_train_model_path)
        print(self.model.test(self_train_test_data_path))
        return self

    @staticmethod
    def _modify_predict_result(predictions):
        def transform_label(text_label):
            return int(text_label[0][-1])

        labels = predictions[0]
        probs = predictions[1]

        modified_predictions = []
        for i in range(len(labels)):
            if transform_label(labels[i]) == 1:
                modified_predictions.append(probs[i][0])

            if transform_label(labels[i]) == 0:
                modified_predictions.append(1 - probs[i][0])

        return modified_predictions

    def predict(self, texts):
        return self._modify_predict_result(self.model.predict(texts))


class TFIDFClassifier(Classifier):

    def __init__(self, tf_idf_config=single_character_tf_idf_config, x=None, y=None):
        self.tf_idf_config = tf_idf_config
        self.x = x
        self.y = y
        self.vectorizer: TfidfVectorizer = None
        self.classifier: LogisticRegression = None

    @staticmethod
    def train_tf_idf_features(text, tf_idf_config):
        vectorizer = TfidfVectorizer(input='content', encoding='utf-8', decode_error='ignore', lowercase=False,
                                     **tf_idf_config)
        vectorizer.fit(text)
        return vectorizer

    def transform_text_to_vector(self, text):
        return self.vectorizer.transform(text)

    def predict(self, texts):
        return self.classifier.predict_proba(self.transform_text_to_vector(texts))[:, 1]

    def train(self):
        self.vectorizer = self.train_tf_idf_features(self.x, self.tf_idf_config)
        vectors = self.transform_text_to_vector(self.x)
        clf = LogisticRegression(random_state=0).fit(vectors, self.y)
        self.classifier = clf
        return self

    def load_model(self):
        pass


class LSTMClassifier(Classifier):

    def load_model(self):
        pass

    def train(self):
        pass

    def predict(self, texts):
        pass


if __name__ == '__main__':
    # data = Sentences.read_train_data()
    # TFIDFClassifier(x=data["sentence"], y=data["label"]).train().evaluate()
    # # print(score)
    # FastTextClassifier(self_train_model_path).evaluate()
    FastTextClassifier().train().evaluate()
