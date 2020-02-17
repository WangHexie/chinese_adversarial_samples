import os
import pickle
import re
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import pkuseg


def root_dir():
    return Path(os.path.abspath(os.path.dirname(__file__))).parent.parent


def read_data(path: tuple):
    return pd.read_csv(os.path.join(root_dir(), *path))


def save_tmp_data(data, path):
    os.makedirs(os.path.join(root_dir(), "models", *path[:-2]), exist_ok=True)
    with open(os.path.join(root_dir(), "models", *path), "wb") as f:
        pickle.dump(data, f)


def read_tmp_data(path):
    with open(os.path.join(root_dir(), "models", *path), "rb") as f:
        data = pickle.load(f)
    return data


class Sentences:

    @staticmethod
    def __read_insult_csv() -> pd.DataFrame:
        path = os.path.join(root_dir(), "data", "dirty_sentence.csv")
        return pd.read_csv(path, names=["sentence", "indirect_label"], index_col=False, encoding="utf-8")

    @staticmethod
    def __read_insult_txt() -> List[str]:
        path = os.path.join(root_dir(), "data", "Insult.txt")
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        return txt.split()

    @staticmethod
    def __remove_emoji(string):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)

        return emoji_pattern.sub(r'', string)

    @staticmethod
    def read_insult_data()->pd.DataFrame:
        insult_df = Sentences.__read_insult_csv()
        insult_txt = Sentences.__read_insult_txt()

        insult_txt_df = pd.DataFrame(np.array([insult_txt, np.ones(len(insult_txt)) * 2]).T, columns=["sentence", "indirect_label"])
        return pd.concat([insult_df, insult_txt_df], ignore_index=True)

    @staticmethod
    def read_positive_data():
        path = os.path.join(root_dir(), "data", "train.txt")
        return pd.read_csv(path, index_col=0, names=["indirect_label", "sentence"])

    @staticmethod
    def read_full_data(num_of_positive=2361, ignore_indirect_data=True):
        negative_data = Sentences.read_insult_data()
        if ignore_indirect_data:
            negative_data = negative_data[negative_data["indirect_label"] != 1]
            negative_data.reset_index(inplace=True)
        negative_data["label"] = np.ones(len(negative_data))

        positive_data = Sentences.read_positive_data()
        positive_data["label"] = np.zeros(len(positive_data))
        positive_data = positive_data.iloc[:num_of_positive]

        full_data = pd.concat([negative_data, positive_data], ignore_index=True, sort=False)[["sentence", "label"]]

        full_data["sentence"] = full_data["sentence"].map(lambda x: Sentences.__remove_emoji(x))

        return full_data.drop_duplicates()


class Tokenizer:
    def __init__(self):
        self.seg = pkuseg.pkuseg()

    def tokenize(self, text):
        return self.seg.cut(text)


if __name__ == '__main__':
    # data = Sentences.read_full_data(ignore_indirect_data=False)
    # print(len(data[data["label"]==1]))
    # data = Sentences.read_insult_data()
    # print(data[data["indirect_label"] == 1])
    print(Tokenizer().tokenize("你说的东西我听不懂"))
