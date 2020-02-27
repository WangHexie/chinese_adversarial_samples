import csv
from typing import List

import numpy as np
import pandas as pd

from src.config.configs import self_train_model_path, SOTAAttackConfig, full_word_tf_idf_config, augment_data_path, \
    self_train_train_data_path, self_train_test_data_path
from src.data.dataset import Sentences
from src.manipulate.importance_based import ImportanceBased
from src.manipulate.rule_based import SimpleDeleteAndReplacement
from src.models.classifier import FastTextClassifier, TFIDFClassifier
from src.predict.word_vector import WordVector


class DataAugment:
    @staticmethod
    def create_data(sentences: List, augment_way_list: List[callable]):
        final_sentences = []
        sentences = pd.Series(sentences)
        for func in augment_way_list:
            final_sentences = final_sentences + sentences.map(func).tolist()

        return final_sentences

    @staticmethod
    def save_data(sentences, path):
        pd.DataFrame(np.array([sentences, np.ones(len(sentences))]).T).to_csv(path, header=False, index=False)

    @staticmethod
    def auto_augment(save_data, save_path):
        data = Sentences.read_train_data()
        path = save_path
        train_data = save_data
        insult_data = train_data[train_data["label"] == 1]["sentence"].tolist()

        sentences = DataAugment.create_data(insult_data,
                                            [SimpleDeleteAndReplacement.replace_dirty_word,

                                             ImportanceBased([
                                                 FastTextClassifier()
                                             ],
                                                 word_vector=WordVector(),
                                                 attack_config=SOTAAttackConfig(num_of_synonyms=30,
                                                                                threshold_of_stopping_attack=0.001,
                                                                                tokenize_method=1,
                                                                                word_use_limit=-1)).craft_one_adversarial_sample,

                                             ImportanceBased([
                                                 TFIDFClassifier(tf_idf_config=full_word_tf_idf_config,
                                                                 x=data["sentence"],
                                                                 y=data["label"]).train()
                                             ],
                                                 word_vector=WordVector(),
                                                 attack_config=SOTAAttackConfig(num_of_synonyms=30,
                                                                                threshold_of_stopping_attack=0.0001,
                                                                                tokenize_method=1,
                                                                                word_use_limit=-1)).craft_one_adversarial_sample,

                                             ImportanceBased([
                                                 FastTextClassifier()
                                             ],
                                                 word_vector=WordVector(),
                                                 attack_config=SOTAAttackConfig(num_of_synonyms=20,
                                                                                threshold_of_stopping_attack=0.001,
                                                                                tokenize_method=1,
                                                                                word_use_limit=20)).craft_one_adversarial_sample,
                                             ])
        negative = pd.DataFrame(np.array([sentences, np.ones(len(sentences)).tolist()]).T, columns=["sentence", "label"])
        negative['label'] = np.ones(len(sentences))
        negative['label'] = negative['label'].map(lambda x:int(x)).astype('int')
        train_data = pd.concat([negative, train_data], ignore_index=True)
        train_data['label'] = train_data['label'].astype('int')
        train_data['label'] = train_data['label'].map(lambda x: "__label__" + str(x))
        train_data.to_csv(path, header=False, index=False, sep=' ',
                          quoting=csv.QUOTE_NONE)


        # DataAugment.save_data(sentences, augment_data_path)


if __name__ == '__main__':
    # DataAugment.auto_augment(save_data=Sentences.read_train_data(), save_path=self_train_train_data_path)
    DataAugment.auto_augment(save_data=Sentences.read_test_data(), save_path=self_train_test_data_path)
