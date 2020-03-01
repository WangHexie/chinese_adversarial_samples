from dataclasses import asdict

from src.config.configs import SOTAAttackConfig, TFIDFConfig
from src.data.dataset import Sentences
from src.manipulate.importance_based import ImportanceBased, ReplacementEnsemble
from src.manipulate.rule_based import PhoneticList, ListOfSynonyms, ReplaceWithPhonetic
from src.models.classifier import TFIDFClassifier, FastTextClassifier
from src.predict.word_vector import WordVector
import numpy as np

def input_function():
    try:
        score = int(input("right:"))
        if score not in [0, 1]:
            raise Exception
        return score
    except Exception:
        print("error")
        return input_function()


def annotation_text():
    data = Sentences.read_full_data()

    config = SOTAAttackConfig(num_of_synonyms=40,
                                       threshold_of_stopping_attack=0.001, tokenize_method=1, word_use_limit=20,
                                       text_modify_percentage=0.5)

    pr = ReplacementEnsemble([
        FastTextClassifier(),
        # FastTextClassifier().train(),
        # TFIDFEmbeddingClassifier(word_vector=WordVector(tencent_embedding_path), tf_idf_config=full_word_tf_idf_config,
        #                          x=data["sentence"],
        #                          y=data["label"]).train(),
        # TFIDFClassifier(x=data["sentence"], y=data["label"]).train(),
        TFIDFClassifier(tf_idf_config=asdict(TFIDFConfig(ngram_range=(1, 3),
                                                         min_df=0.0005)), x=data["sentence"],
                        y=data["label"]).train()
    ],
        word_vector=WordVector(),
        attack_config=config,
        replacement_classes=[ListOfSynonyms(word_vector=WordVector(), attack_config=config)]
    )
    r = ReplaceWithPhonetic()

    data = Sentences.read_insult_data().sample(frac=1).reset_index(drop=True).iloc[:500]

    p = data["sentence"].map(lambda x: r.replace(x))

    num_of_data_to_test = 30
    scores = np.array(pr.predict_use_classifiers(p.values.tolist())) < 0.5
    p = p[scores]

    data_to_test = p.values.tolist()[:num_of_data_to_test]
    success_number = 0
    for i in range(num_of_data_to_test):
        print(data_to_test[i])
        success_number += input_function()

    print("success rate:", success_number / num_of_data_to_test)

    # print(p.values.tolist()[:num_of_data_to_test])
    print(data["sentence"].values.tolist()[:num_of_data_to_test])


if __name__ == '__main__':
    annotation_text()