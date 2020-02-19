import numpy as np

from src.config.configs import no_chinese_tokenizer_word_tf_idf_config, single_character_tf_idf_config, \
    SOTAAttackConfig, strong_attack_config, self_train_model_path, tencent_embedding_path
from src.data.dataset import Sentences, Tokenizer
from src.models.classifier import FastTextClassifier
from src.predict.word_vector import WordVector
from src.train.identify_importance_word import InspectFeatures

dirty_character_list = InspectFeatures(single_character_tf_idf_config).locate_top_dirty_character(70)
dirty_word_list = InspectFeatures.is_dirty_by_classifier(FastTextClassifier(),
                                                         InspectFeatures(
                                                             no_chinese_tokenizer_word_tf_idf_config).locate_top_dirty_character(
                                                             1000),
                                                         0.1)


def insert_sting_middle(string, word, index):
    return string[:index] + word + string[index:]


def tokenizer_selector(method) -> callable:
    if method == 0:
        return list

    if method == 1:
        return Tokenizer().tokenize


class PaperRealize:

    def __init__(self, classifier, word_vector, attack_config: SOTAAttackConfig):
        self.classifier = classifier
        self.word_vector = word_vector
        self.attack_config = attack_config
        self.tokenizer = tokenizer_selector(method=self.attack_config.tokenize_method)

    def tokenize_text(self, text):
        return self.tokenizer(text)

    def _identify_important_word(self, text_list):
        # todo: add stop word list
        leave_one_text_list = ["".join(text_list)]

        for i in range(len(text_list)):
            text = text_list.copy()
            # act as out-of-vocabulary
            text[i] = '叇'
            leave_one_text_list.append("".join(text))

        probs = self.classifier.predict(leave_one_text_list)
        origin_prob = probs[0]
        leave_one_probs = probs[1:]
        return origin_prob - np.array(leave_one_probs)

    def _temporal_head_score(self, text_list):
        top_n = [''.join(text_list[:i+1]) for i in range(len(text_list))]
        probs = self.classifier.predict(top_n)

    def _replace_text_and_predict(self, text_tokens, synonyms, index):
        sentences = []
        for word in synonyms:
            temp = text_tokens.copy()
            temp[index] = word
            sentences.append("".join(temp))
        return self.classifier.predict(sentences)

    @staticmethod
    def _choose_synonym_to_replace(scores, original_scores, score_threshold=0.4):
        """
        Todo: choose a real one
        :param scores:
        :param original_scores:
        :param score_threshold:
        :return: index_of synonym , control character
        """
        if min(scores) > original_scores:
            return None, None

        if min(scores) < score_threshold:
            return np.argmin(scores), True
        return np.argmin(scores), False

    @staticmethod
    def _replace_with_synonym(text_token, index_to_replace, synonyms, index_of_synonyms):
        if index_of_synonyms is None:
            return text_token
        text_token[index_to_replace] = synonyms[index_of_synonyms]
        return text_token

    def craft_one_adversarial_sample(self, text):
        origin_score = self.classifier.predict([text])[0]

        # tokenize text
        tokenized_text = self.tokenize_text(text)

        # identify important word
        important_score = self._identify_important_word(tokenized_text)

        important_word_index = np.argsort(important_score)[::-1]

        for i in range(min(len(tokenized_text), self.attack_config.top_num_of_word_to_modify)):
            # find synonyms of important word

            index_of_word_to_replace = important_word_index[i]

            try:
                synonyms = self.word_vector.most_similar(tokenized_text[index_of_word_to_replace],
                                                         topn=self.attack_config.num_of_synonyms)
            except KeyError:
                continue
            # replace and predict
            scores = self._replace_text_and_predict(tokenized_text, synonyms, index_of_word_to_replace)

            synonym_index, has_succeeded = self._choose_synonym_to_replace(scores, origin_score)
            tokenized_text = self._replace_with_synonym(tokenized_text, index_of_word_to_replace, synonyms,
                                                        synonym_index)
            if has_succeeded:
                break

        return "".join(tokenized_text)


class SelfDefined(PaperRealize):
    def _identify_important_word(self, text_list):
        word_to_replace = ''
        # todo: add stop word list
        leave_one_text_list = ["".join(text_list)]

        for i in range(len(text_list)):
            text = text_list.copy()
            # act as out-of-vocabulary
            text[i] = word_to_replace
            leave_one_text_list.append("".join(text))

        probs = self.classifier.predict(leave_one_text_list)
        origin_prob = probs[0]
        leave_one_probs = probs[1:]
        return origin_prob - np.array(leave_one_probs)

    def craft_one_adversarial_sample(self, text):
        origin_score = self.classifier.predict([text])[0]

        word_to_replace = ['']

        # tokenize text
        tokenized_text = self.tokenize_text(text)

        # identify important word
        important_score = self._identify_important_word(tokenized_text)
        important_word_index = np.argsort(important_score)[::-1]

        for i in range(min(len(tokenized_text), self.attack_config.top_num_of_word_to_modify)):
            # find synonyms of important word

            index_of_word_to_replace = important_word_index[i]

            synonyms = word_to_replace

            # replace and predict
            scores = self._replace_text_and_predict(tokenized_text, synonyms, index_of_word_to_replace)

            synonym_index, has_succeeded = self._choose_synonym_to_replace(scores, origin_score)
            tokenized_text = self._replace_with_synonym(tokenized_text, index_of_word_to_replace, synonyms,
                                                        synonym_index)
            if has_succeeded:
                break

        return "".join(tokenized_text)


def replace_dirty_word(sentences):
    global dirty_character_list
    global dirty_word_list

    # for k in dirty_word_list:
    #     sentences = sentences.replace(k, insert_sting_middle(k, random.choice(punctuation), 1))
    #
    # for k in dirty_character_list:
    #     sentences = sentences.replace(k, lazy_pinyin(k)[0])

    for k in dirty_word_list:
        sentences = sentences.replace(k, '')

    for k in dirty_character_list:
        sentences = sentences.replace(k, '')

    return sentences


if __name__ == '__main__':
    pr = PaperRealize(FastTextClassifier(self_train_model_path),
                      word_vector=WordVector(tencent_embedding_path),
                      attack_config=strong_attack_config)
    data = Sentences.read_insult_data()

    print(dirty_character_list)
    print(dirty_word_list)
    p = data["sentence"].map(lambda x: pr.craft_one_adversarial_sample(x))
    # scores = pr.classifier.predict(p.values.tolist())

    print(len(InspectFeatures.is_dirty_by_classifier(pr.classifier, dirty_character_list, 0.4)))
    print(len(InspectFeatures.is_dirty_by_classifier(pr.classifier, dirty_word_list, 0.4)))
    print(p.values.tolist()[:30])
    print(data["sentence"].values.tolist()[:30])
