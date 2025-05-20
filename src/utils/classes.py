import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from Mylib import myfuncs
import numpy as np
import pandas as pd


class BleuScoreCustomMetric(tf.keras.metrics.Metric):
    def __init__(self, name="bleu", **kwargs):
        super().__init__(name=name, **kwargs)
        self.list_bleu = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)  # Convert về giống shape của y_true

        list_bleu_on_1batch = self.get_bleu(y_true, y_pred)
        self.list_bleu.append(list_bleu_on_1batch)

    def result(self):  # Tính toán vào cuối epoch
        return tf.reduce_mean(self.list_bleu)

    def reset_state(self):  # Trước khi vào epoch mới
        self.list_bleu = []

    def get_values_g1(self, tensor):
        filtered = tf.boolean_mask(tensor, tensor > 1)
        filtered = filtered.numpy().astype("str").tolist()
        return filtered

    def get_bleu(self, y_true, y_pred):
        words_in_true = [self.get_values_g1(item) for item in y_true]
        words_in_pred = [self.get_values_g1(item) for item in y_pred]

        smooth = SmoothingFunction()
        list_bleu = [
            sentence_bleu([ref], pred, smoothing_function=smooth.method1)
            for ref, pred in zip(words_in_true, words_in_pred)
        ]

        return list_bleu


class CustomisedModelCheckpoint(tf.keras.callbacks.Callback):
    """Callback để lưu model tốt nhất theo từng epoch

    Attributes:
        filepath (str): đường dẫn đến best model
        scoring_path (str): đường dẫn lưu các scoring ứng với best model tìm được
        monitor (str): chỉ số đánh giá (đánh giá theo **val**), *vd:* accuracy, loss, bleu, ....
        indicator (str): chỉ tiêu



    Examples:
        Với **monitor = val_accuracy và indicator = 0.99**

        Tìm model thỏa val_accuracy > 0.99 và train_accuracy > 0.99 (1) và val_accuracy là lớn nhất trong số đó

        Nếu không thỏa (1) thì lấy theo val_accuracy lớn nhất

    """

    def __init__(
        self, filepath: str, scoring_path: str, monitor: str, indicator: float
    ):
        super().__init__()
        self.filepath = filepath
        self.scoring_path = scoring_path
        self.monitor = monitor
        self.indicator = indicator

    def on_train_begin(self, logs=None):
        self.sign_for_score = (
            1  # Nếu scoring là loss thì lấy âm -> quy về tìm lớn nhất thôi
        )
        if (
            self.monitor.endswith("loss")
            or self.monitor.endswith("mse")
            or self.monitor.endswith("mae")
        ):
            self.indicator = -self.indicator
            self.sign_for_score = -1

        self.per_epoch_val_scores = []
        self.per_epoch_train_scores = []
        self.models = []

    def on_epoch_end(self, epoch, logs=None):
        self.models.append(self.model)
        self.per_epoch_val_scores.append(
            logs.get(f"val_{self.monitor}") * self.sign_for_score
        )
        self.per_epoch_train_scores.append(logs.get(self.monitor) * self.sign_for_score)

    def on_train_end(self, logs=None):
        # Tìm model tốt nhất
        self.per_epoch_val_scores = np.asarray(self.per_epoch_val_scores)
        self.per_epoch_train_scores = np.asarray(self.per_epoch_train_scores)

        # Tìm các model thỏa train_scoring, val_scoring > target (đề ra)
        indexs_good_model = np.where(
            (self.per_epoch_val_scores > self.indicator)
            & (self.per_epoch_train_scores > self.indicator)
        )[0]

        # Tìm model tốt nhất
        index_best_model = None
        if (
            len(indexs_good_model) == 0
        ):  # Nếu ko có model nào đạt chỉ tiêu thì lấy cái tốt nhất
            index_best_model = np.argmax(self.per_epoch_val_scores)
        else:
            val_series = pd.Series(
                self.per_epoch_val_scores[indexs_good_model], index=indexs_good_model
            )
            index_best_model = val_series.idxmax()

        # TODO: d
        print(f"index best model thông qua modelcheckpoint = {index_best_model}")
        # d

        best_model = self.models[index_best_model]
        best_model_train_scoring = self.per_epoch_train_scores[index_best_model]
        best_model_val_scoring = self.per_epoch_val_scores[index_best_model]

        # Lưu model tốt nhất và train,val scoring tương ứng
        best_model.save(self.filepath)
        myfuncs.save_python_object(
            self.scoring_path, (best_model_train_scoring, best_model_val_scoring)
        )
