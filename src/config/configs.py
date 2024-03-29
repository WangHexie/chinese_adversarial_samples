import os
from dataclasses import dataclass, asdict
from typing import Callable, List

import pkuseg

from src.data.basic_functions import root_dir


@dataclass
class TFIDFConfig:
    ngram_range: tuple = (1, 1)
    preprocessor = None
    tokenizer: Callable = None
    analyzer: str = 'char'
    stop_words: List = None
    max_df: float = 0.8
    min_df: float = 0.01
    max_features: int = None
    vocabulary: List = None
    binary: bool = False
    norm: str = 'l2'
    use_idf: bool = True
    smooth_idf: bool = True
    sublinear_tf: bool = False


single_character_tf_idf_config = asdict(TFIDFConfig(min_df=0.0005))
no_chinese_tokenizer_word_tf_idf_config = asdict(TFIDFConfig(ngram_range=(2, 3), min_df=0.0005))
full_tokenizer_word_tf_idf_config = asdict(TFIDFConfig(ngram_range=(1, 3), min_df=0.0005))  # not using tokenizer only ngram
full_word_tf_idf_config = asdict(TFIDFConfig(analyzer="word", tokenizer=pkuseg.pkuseg().cut, min_df=0.0))


@dataclass
class SOTAAttackConfig:
    top_num_of_word_to_modify: int = 5
    num_of_synonyms: int = 5
    threshold_of_stopping_attack: float = 0.4
    tokenize_method: int = 0
    text_modify_percentage: float = 0.5
    word_use_limit: int = 20
    word_to_replace: str = '叇'
    importance_function: int = 1


@dataclass
class DeepModelConfig:
    input_length:int = 80
    verbose:int = 0


strong_attack_config = SOTAAttackConfig(num_of_synonyms=20,
                                        threshold_of_stopping_attack=0.00001, tokenize_method=1)

tencent_embedding_path = os.path.join(root_dir(), "models", "small_Tencent_AILab_ChineseEmbedding.txt")
weibo_embedding_path = os.path.join(root_dir(), "models", "sgns.weibo.word.bz2")
self_train_model_path = os.path.join(root_dir(), "models", "self_train.bin")
self_train_test_data_path = os.path.join(root_dir(), "data", "test.csv")
self_train_train_data_path = os.path.join(root_dir(), "data", "train.csv")
augment_data_path = os.path.join(root_dir(), "data", "augment_data.csv")
deep_model_path = os.path.join(root_dir(), "models", "deep_model.h5")

if __name__ == '__main__':
    print(self_train_test_data_path)
    print(no_chinese_tokenizer_word_tf_idf_config)
