import os

import fasttext

from src.data.dataset import root_dir


class FastTextClassifier:
    def __init__(self, model_path=os.path.join(root_dir(), "models", "mini.ftz")):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        model = fasttext.load_model(self.model_path)
        self.model = model
        return self

    @staticmethod
    def _modify_predict_result(predictions):
        def transform_label(text_label):
            return int(text_label[0][-1])

        labels = predictions[0]
        probs = predictions[1]

        modified_predictions = []
        for i in range(len(labels)):
            if transform_label(labels[i]) == 1:
                modified_predictions.append(probs[i][0])

            if transform_label(labels[i]) == 0:
                modified_predictions.append(1 - probs[i][0])

        return modified_predictions

    def predict(self, texts):
        return self._modify_predict_result(self.model.predict(texts))


if __name__ == '__main__':
    print(FastTextClassifier().load_model().predict(["你爸吃屎", "你爸吃shi"]))