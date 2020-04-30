from dataclasses import asdict

import numpy as np

from src.config.configs import SOTAAttackConfig, TFIDFConfig, full_word_tf_idf_config, self_train_model_path, \
    DeepModelConfig
from src.data.dataset import Sentences
from src.data.measure_distance import DistanceCalculator
from src.embedding.word_vector import WordVector
from src.manipulate.importance_based import ReplacementEnsemble
from src.manipulate.rule_based import DeleteDirtyWordFoundByTokenizer, \
    InsertSpace, \
    ReplaceWithPhonetic, \
    InsertPunctuationInDirtyWordFoundByTokenizer, append_word, ListOfSynonyms
from src.models.classifier import FastTextClassifier, TFIDFClassifier, EmbeddingSVM, EmbeddingLGBM, DeepModel
from src.models.deep_model import SimpleCnn, SimpleRNN


class EvaluateAttack:
    @staticmethod
    def choose_dataset(data_type=0):
        """

        :param data_type: 0: all insult data. 1:test data
        :return:
        """
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
        """

        :param attack_method:
        :param classifier:
        :param dataset_type: 0: all insult data. 1:test data
        :return:
        """
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


def self_defined_function(manipulate_func: callable, classifiers):
    control_group = {"delete all dirty word": DeleteDirtyWordFoundByTokenizer().replace}

    for key, item in classifiers.items():
        print("-----------start {} evaluate---------------".format(key))
        EvaluateAttack.evaluate(manipulate_func, item, dataset_type=1)

    # for name, func in control_group.items():
    #     for cls_name, cls in classifiers.items():
    #         print("-----------control group {}, {} evaluate---------------".format(name, cls_name))
    #         EvaluateAttack.evaluate(func, cls, dataset_type=1)


data = Sentences.read_train_data()
word_vector = WordVector()
classifiers = {
    "tfidf ngram": TFIDFClassifier(tf_idf_config=asdict(TFIDFConfig(ngram_range=(1, 3), min_df=0.0005)),
                                   x=data["sentence"],
                                   y=data["label"]).train(),
    "tfidf tokenizer": TFIDFClassifier(tf_idf_config=full_word_tf_idf_config,
                                       x=data["sentence"],
                                       y=data["label"]).train(),
    "embedding svn": EmbeddingSVM(word_vector=word_vector, tf_idf_config=full_word_tf_idf_config).train(
        x=data["sentence"], y=data["label"]),

    "embedding lgbm": EmbeddingLGBM(x=data["sentence"], y=data["label"], word_vector=word_vector,
                                    tf_idf_config=full_word_tf_idf_config).train(),
    "fasttext provided": FastTextClassifier(),
    "fasttext self-trained": FastTextClassifier(self_train_model_path),
    "cnn": DeepModel(word_vector=word_vector, config=DeepModelConfig(), tokenizer=list, model_creator=SimpleCnn).train(
        x=data["sentence"].values, y=data["label"].values),
    "rnn": DeepModel(word_vector=word_vector, config=DeepModelConfig(), tokenizer=list,
                     model_creator=SimpleRNN).train(x=data["sentence"].values, y=data["label"].values),

}


def simple_evaluation():
    import warnings

    warnings.simplefilter("ignore")
    modi = {
        "insert space": InsertSpace().replace,
        "insert space in dirty word": InsertPunctuationInDirtyWordFoundByTokenizer().replace,
        "replace with syn": ReplaceWithPhonetic().replace,
        "append good word": append_word,
        "delete all": DeleteDirtyWordFoundByTokenizer().replace
    }
    for k, i in modi.items():
        print(k)
        self_defined_function(i, classifiers)


def sota_evaluation():
    import warnings

    warnings.simplefilter("ignore")

    re_config = SOTAAttackConfig(num_of_synonyms=40,
                                 threshold_of_stopping_attack=0.1, tokenize_method=1, word_use_limit=10000,
                                 text_modify_percentage=0.5, word_to_replace='', importance_function=0)

    re_config2 = SOTAAttackConfig(num_of_synonyms=40,
                                 threshold_of_stopping_attack=0.1, tokenize_method=1, word_use_limit=10000,
                                 text_modify_percentage=0.5, word_to_replace='', importance_function=1)

    modi = {
        "all delete": ReplacementEnsemble([
            classifiers["rnn"],
        ],
            word_vector=word_vector,
            attack_config=re_config,
            replacement_classes=[ListOfSynonyms(word_vector=word_vector, attack_config=re_config)],
            classifier_coefficient=[1]
        ).craft_one_adversarial_sample,
        "delete ensemble": ReplacementEnsemble([
            classifiers["rnn"],
        ],
            word_vector=word_vector,
            attack_config=re_config2,
            replacement_classes=[ListOfSynonyms(word_vector=word_vector, attack_config=re_config)],
            classifier_coefficient=[1]
        ).craft_one_adversarial_sample,

    }

    classifiers.pop("cnn")
    classifiers.pop("tfidf tokenizer")
    classifiers.pop("embedding svn")
    classifiers.pop("fasttext provided")
    classifiers.pop("fasttext self-trained")
    classifiers.pop("rnn")

    for k, i in modi.items():
        print(k)
        self_defined_function(i, classifiers)


def sota_evaluation2():
    import warnings

    warnings.simplefilter("ignore")

    classifiers.pop("tfidf tokenizer")
    classifiers.pop("embedding svn")
    classifiers.pop("fasttext provided")
    classifiers.pop("fasttext self-trained")

    for k, i in classifiers.items():
        print("***************************************************")
        print(k)

        re_config = SOTAAttackConfig(num_of_synonyms=40,
                                     threshold_of_stopping_attack=0.1, tokenize_method=1, word_use_limit=10000,
                                     text_modify_percentage=0.5, word_to_replace='', importance_function=0)

        modi_f = ReplacementEnsemble([
            i,
        ],
            word_vector=word_vector,
            attack_config=re_config,
            replacement_classes=[ListOfSynonyms(word_vector=word_vector, attack_config=re_config)],
            classifier_coefficient=[1]
        ).craft_one_adversarial_sample

        temp = classifiers.pop(k)

        self_defined_function(modi_f, classifiers)
        classifiers[k] = temp


if __name__ == '__main__':
    data = Sentences.read_train_data()

    sota_evaluation2()

    # word_vector = WordVector()
    #

    # #
    # func = DeleteDirtyWordFoundByTokenizer()
    # p = ReplaceWithPhoneticNoSpecialWord()

    # def repp(string):
    #     return p.replace(func.replace(string))
    # print(rep.craft_one_adversarial_sample("傻逼呆湾"))

    # self_defined_function(InsertSpace().replace)
    # ListOfSynonyms(word_vector=WordVector(), attack_config=SOTAAttackConfig(num_of_synonyms=40,
    #                                                                         threshold_of_stopping_attack=0.001,
    #                                                                         tokenize_method=1,
    #                                                                         word_use_limit=20,
    #                                                                         text_modify_percentage=0.5))
