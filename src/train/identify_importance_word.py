import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config.configs import single_character_tf_idf_config, no_chinese_tokenizer_word_tf_idf_config
from src.data.dataset import Sentences


class InspectFeatures:
    def __init__(self, tf_idf_config=single_character_tf_idf_config):
        self.tf_idf_config = tf_idf_config

    @staticmethod
    def tf_idf_features(text, tf_idf_config):
        vectorizer = TfidfVectorizer(input='content', encoding='utf-8', decode_error='ignore', lowercase=False,
                                     **tf_idf_config)
        text_features = vectorizer.fit_transform(text)
        feature_names = vectorizer.get_feature_names()
        print("identified feature length:", len(feature_names))
        return text_features, feature_names

    @staticmethod
    def find_important_features_by_using_linear_model(features, labels):
        clf = LogisticRegression(random_state=0).fit(features, labels)
        print("classifier score:", clf.score(features, labels))
        return clf.coef_

    @staticmethod
    def is_dirty_by_classifier(classifier, dirty_list, threshold=0.4):
        scores = classifier.predict(dirty_list)
        is_dirty = np.array(scores) > threshold

        not_dirty = [dirty_list[i] for i in range(len(scores)) if not is_dirty[i]]

        print("not dirty:", len(not_dirty), not_dirty)
        return [dirty_list[i] for i in range(len(scores)) if is_dirty[i]]

    def locate_top_dirty_character(self, number_of_dirty=70):
        data = Sentences.read_full_data()
        text_features, feature_names = InspectFeatures.tf_idf_features(data["sentence"], self.tf_idf_config)
        coef = InspectFeatures.find_important_features_by_using_linear_model(text_features, data["label"])[0]
        dirty_word_index = np.argsort(coef)[::-1]
        return [feature_names[dirty_word_index[i]] for i in range(len(coef)) if coef[dirty_word_index[i]] > 0][:number_of_dirty]


if __name__ == '__main__':
    print(InspectFeatures(no_chinese_tokenizer_word_tf_idf_config).locate_top_dirty_character())
