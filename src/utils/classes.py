import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from Mylib import myfuncs, tf_myfuncs
import numpy as np
import pandas as pd
from src.utils import funcs


class BleuScoreCustomMetric(tf.keras.metrics.Metric):
    def __init__(self, name="bleu", **kwargs):
        super().__init__(name=name, **kwargs)
        self.list_bleu = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)  # Convert về giống shape của y_true

        list_bleu_on_1batch = funcs.get_list_bleu_for_true_pred(y_true, y_pred)
        self.list_bleu += (
            list_bleu_on_1batch  # Thêm batch mới vào kết quả cuối cùng của epoch
        )

    def result(self):  # Tính toán vào cuối epoch
        return tf.reduce_mean(self.list_bleu)

    def reset_state(self):  # Trước khi vào epoch mới
        self.list_bleu = []


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


class MachineTranslationEvaluator:
    """Dùng để đánh giá tổng quát bài toán Dịch Máy <br>
    Đánh giá chỉ số BLEU

    Args:
        model (_type_): _description_
        train_ds (_type_): _description_
        val_ds (_type_, optional): Nếu None thì chỉ đánh giá trên 1 tập thôi (tập test). Defaults to None.
    """

    def __init__(self, model, train_ds, val_ds=None):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds

    def evaluate_train_classifier(self):
        # Get thực tế và dự đoán
        train_target, train_pred = tf_myfuncs.get_full_target_and_pred_for_DLmodel(
            self.model, self.train_ds
        )
        val_target, val_pred = tf_myfuncs.get_full_target_and_pred_for_DLmodel(
            self.model, self.val_ds
        )

        # Đánh giá: bleu + ...
        train_bleu = np.mean(
            funcs.get_list_bleu_for_true_pred(train_target, train_pred)
        )  # Lấy trung bình
        val_bleu = np.mean(funcs.get_list_bleu_for_true_pred(val_target, val_pred))

        result = f"Train BLEU: {train_bleu}\n"
        result += f"Val BLEU: {val_bleu}\n"

        return result

    def evaluate_test_classifier(self):
        # Get thực tế và dự đoán
        test_target, test_pred = tf_myfuncs.get_full_target_and_pred_for_DLmodel(
            self.model, self.train_ds
        )

        # Đánh giá: bleu + ...
        test_bleu = np.mean(funcs.get_list_bleu_for_true_pred(test_target, test_pred))

        result = f"Test BLEU: {test_bleu}\n"

        return result

    def evaluate(self):
        return (
            self.evaluate_train_classifier()
            if self.val_ds is not None
            else self.evaluate_test_classifier()
        )
