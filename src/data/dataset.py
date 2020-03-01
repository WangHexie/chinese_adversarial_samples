import csv
import os
import pickle
import re
from typing import List

import numpy as np
import pandas as pd
import pkuseg
from sklearn.model_selection import train_test_split

from src.config.configs import self_train_test_data_path, self_train_train_data_path, augment_data_path
from src.data.basic_functions import root_dir


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
        return txt.split('\n')

    @staticmethod
    def __read_train_insult_data() -> pd.DataFrame:
        path = '/tcdata/benchmark_texts.txt'
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        sentences = txt.split('\n')
        insult_txt_df = pd.DataFrame(np.array([sentences, np.zeros(len(sentences))]).T,
                                     columns=["sentence", "indirect_label"])

        return insult_txt_df

    @staticmethod
    def __remove_emoji(string):
        REGEX_TO_REMOVE = re.compile(
            r"[^\u4E00-\u9FA5a-zA-Z0-9\!\@\#\$\%\^\&\*\(\)\-\_\+\=\`\~\\\|\[\]\{\}\:\;\"\'\,\<\.\>\/\?\ \t，。！？]")
        # emoji_pattern = re.compile("["
        #                            u"\U0001F600-\U0001F64F"  # emoticons
        #                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        #                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
        #                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        #                            "]+", flags=re.UNICODE)
        return REGEX_TO_REMOVE.sub(r'', string)

    @staticmethod
    def _remove_quotation_and_others(text):
        return re.sub(r'[\"]', '', text)

    @staticmethod
    def _remove_white_space(text):
        return text.replace(" ", "")

    @staticmethod
    def _remove_hash(text):
        return re.sub(r'\{\%\#.*\#\%\}', '', text)

    @staticmethod
    def _remove_expression(text):
        return re.sub(r'\[.*\]', '', text)

    @staticmethod
    def _remove_weibo_video(text):
        return re.sub(r'\{\%.*\%\}', '', text)

    @staticmethod
    def _remove_weibo_style(text):
        return Sentences._remove_weibo_video(Sentences._remove_expression(Sentences._remove_hash(text)))

    @staticmethod
    def read_insult_data() -> pd.DataFrame:
        insult_df = Sentences.__read_insult_csv()
        insult_txt = Sentences.__read_insult_txt()

        insult_txt_df = pd.DataFrame(np.array([insult_txt, np.ones(len(insult_txt)) * 2]).T,
                                     columns=["sentence", "indirect_label"])

        try:
            train_insult = Sentences.__read_train_insult_data()
            full_data = pd.concat([insult_df, insult_txt_df, train_insult], ignore_index=True)
        except FileNotFoundError:
            print("not in train mode")
            full_data = pd.concat([insult_df, insult_txt_df], ignore_index=True)

        full_data["sentence"] = full_data["sentence"].map(lambda x: Sentences._remove_new_line_symbol(x))

        return full_data

    @staticmethod
    def read_augment_data():
        return pd.read_csv(augment_data_path, names=["sentence", "label"], index_col=False, )

    @staticmethod
    def read_positive_data():
        path = os.path.join(root_dir(), "data", "train.txt")
        return pd.read_csv(path, index_col=0, names=["indirect_label", "sentence"])

    @staticmethod
    def _remove_new_line_symbol(text):
        return text.replace('\n', '')

    @staticmethod
    def read_full_data(num_of_positive="auto", ignore_indirect_data=True) -> pd.DataFrame:
        """

        :param num_of_positive: auto: balance negative data and positive data
        :param ignore_indirect_data:
        :return:
        """
        negative_data = Sentences.read_insult_data()
        if ignore_indirect_data:
            negative_data = negative_data[negative_data["indirect_label"] != 1]
            negative_data.reset_index(inplace=True)
        negative_data["label"] = np.ones(len(negative_data))

        if num_of_positive == "auto":
            num_of_positive = len(negative_data)

        positive_data = Sentences.read_positive_data()
        positive_data["label"] = np.zeros(len(positive_data))
        positive_data = positive_data.iloc[:num_of_positive]
        try:
            augment_data = Sentences.read_augment_data()
            full_data = pd.concat([negative_data, positive_data, augment_data], ignore_index=True, sort=False)[
                ["sentence", "label"]]
        except FileNotFoundError:
            full_data = pd.concat([negative_data, positive_data], ignore_index=True, sort=False)[["sentence", "label"]]

        full_data = full_data.dropna()
        full_data["sentence"] = full_data["sentence"].map(
            lambda x: Sentences._remove_weibo_style(
                Sentences._remove_white_space(Sentences._remove_new_line_symbol(Sentences.__remove_emoji(x)))))
        full_data["label"] = full_data["label"].astype('int')

        return full_data.drop_duplicates(subset="sentence").reset_index()

    def save_train_data(self, num_of_positive="auto", test_data_size = 0.4):
        data = self.read_full_data(num_of_positive=num_of_positive, ignore_indirect_data=True)[["label", "sentence"]]
        data['label'] = data['label'].map(lambda x: "__label__" + str(x))
        data['sentence'] = data['sentence'].astype('str')

        train_data, test_data = train_test_split(data, test_size=test_data_size)
        train_data.to_csv(self_train_train_data_path, header=False, index=False, sep=' ',
                          quoting=csv.QUOTE_NONE)
        test_data.to_csv(self_train_test_data_path, header=False, index=False, sep=' ', quoting=csv.QUOTE_NONE)
        print(test_data)

    @staticmethod
    def read_df_data(path):
        test_data = pd.read_csv(path, names=["label", "sentence"], index_col=False, encoding="utf-8", sep=' ')
        test_data['label'] = test_data['label'].map(lambda x: int(x[-1]))
        test_data["label"] = test_data["label"].astype('int')
        return test_data.dropna()

    @staticmethod
    def read_test_data():
        return Sentences.read_df_data(self_train_test_data_path)

    @staticmethod
    def read_train_data():
        return Sentences.read_df_data(self_train_train_data_path)

    @staticmethod
    def read_similar_word():
        with open(os.path.join(root_dir(), "data", "similar_word.txt"), "r", encoding="gbk") as f:
            words = f.read()

        return Sentences.__remove_emoji(words.replace('\n', '').replace(' ', ''))

    @staticmethod
    def read_common_words():
        with open(os.path.join(root_dir(), "data", "common_words.txt"), "r", encoding="utf-8") as f:
            words = f.read()

        return Sentences.__remove_emoji(words.replace('\n', '').replace(' ', ''))

    @staticmethod
    def read_stop_words():
        with open(os.path.join(root_dir(), "data", "stop_word.txt"), "r", encoding="utf-8") as f:
            words = f.read()

        return list(words.split('\n'))


class Tokenizer:
    def __init__(self):
        self.seg = pkuseg.pkuseg()

    def __call__(self, text):
        return self.seg.cut(text)

    def tokenize(self, text):
        return self.seg.cut(text)


if __name__ == '__main__':
    # data = Sentences.read_full_data(ignore_indirect_data=False)
    print(Sentences().read_stop_words())
    # print(Sentences().save_train_data(-1))
    # print(Sentences.read_common_words())
    # print(Sentences.read_similar_word())
    # print(data)
    # print(len(data[data["label"]==1]))
    # data = Sentences.read_insult_data()
    # print(data[data["indirect_label"] == 1])
    # print(Tokenizer().tokenize("你说的东西我听不懂"))
