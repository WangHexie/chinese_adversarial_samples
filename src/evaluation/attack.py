import numpy as np

from src.config.configs import strong_attack_config, self_train_model_path, tencent_embedding_path, \
    no_chinese_tokenizer_word_tf_idf_config, SOTAAttackConfig, full_word_tf_idf_config
from src.data.dataset import Sentences
from src.data.measure_distance import DistanceCalculator
from src.manipulate.black_box import ImportanceBased, \
    SimpleDeleteAndReplacement, RemoveImportantWord
from src.models.classifier import FastTextClassifier, TFIDFClassifier
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
        print("final_score:", np.dot(distances['final_similarity_score'], success_status) / len(data))

def self_defined_function():
    data = Sentences.read_test_data()

    cls = TFIDFClassifier(x=data["sentence"], y=data["label"]).train()
    pr = ImportanceBased([
        # FastTextClassifier(self_train_model_path),
        # cls,
        TFIDFClassifier(tf_idf_config=full_word_tf_idf_config, x=data["sentence"],
                        y=data["label"]).train()
    ],
        word_vector=WordVector(tencent_embedding_path),
        attack_config=SOTAAttackConfig(num_of_synonyms=40,
                                        threshold_of_stopping_attack=0.01, tokenize_method=1))
    print("-----------start tfidf evaluate---------------")
    EvaluateAttack.evaluate(pr.craft_one_adversarial_sample, cls, dataset_type=1)
    print("-----------fasttext evaluate---------------")
    EvaluateAttack.evaluate(pr.craft_one_adversarial_sample, FastTextClassifier(), dataset_type=1)
    print("-----------fasttext in model evaluate---------------")
    EvaluateAttack.evaluate(pr.craft_one_adversarial_sample, pr.classifiers[0], dataset_type=1)
    print("-----------replace evaluate---------------")
    EvaluateAttack.evaluate(SimpleDeleteAndReplacement.replace_dirty_word, pr.classifiers[0], dataset_type=1)
    print("-----------append fastext evaluate---------------")
    EvaluateAttack.evaluate(SimpleDeleteAndReplacement.random_append_good_word, FastTextClassifier(), dataset_type=1)
    print("-----------append tfidf evaluate---------------")
    EvaluateAttack.evaluate(SimpleDeleteAndReplacement.random_append_good_word, cls, dataset_type=1)

if __name__ == '__main__':
    pass
