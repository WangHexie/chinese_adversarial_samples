from dataclasses import dataclass, asdict
from typing import Callable, List


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


single_character_tf_idf_config = asdict(TFIDFConfig())
no_chinese_tokenizer_word_tf_idf_config = asdict(TFIDFConfig(ngram_range=(2, 3)))


@dataclass
class SOTAAttackConfig:
    top_num_of_word_to_modify: int = 5
    num_of_synonyms: int = 5
    threshold_of_stopping_attack: float = 0.4
    tokenize_method: int = 0


strong_attack_config = SOTAAttackConfig(top_num_of_word_to_modify=6, num_of_synonyms=8,
                                        threshold_of_stopping_attack=0.02, tokenize_method=0)

if __name__ == '__main__':
    print(no_chinese_tokenizer_word_tf_idf_config)
