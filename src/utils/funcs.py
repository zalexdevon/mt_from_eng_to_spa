from src.utils import classes
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import tensorflow as tf
from Mylib import tf_myfuncs


def get_metric_object_by_name(name):
    if name == "bleu":
        return classes.BleuScoreCustomMetric()

    raise ValueError("Chỉ mới định nghĩa cho bleu")


def get_values_g1(tensor):
    filtered = tf.boolean_mask(tensor, tensor > 1)
    filtered = filtered.numpy().astype("str").tolist()
    return filtered


def get_list_bleu_for_true_pred(y_true, y_pred):
    words_in_true = [get_values_g1(item) for item in y_true]
    words_in_pred = [get_values_g1(item) for item in y_pred]

    smooth = SmoothingFunction()
    list_bleu = [
        sentence_bleu([ref], pred, smoothing_function=smooth.method1)
        for ref, pred in zip(words_in_true, words_in_pred)
    ]

    return list_bleu


def save_text_vectorization(text_vectorization, file_path):
    vectorizer_model = tf.keras.Sequential([text_vectorization])
    vectorizer_model.save(file_path)


def load_text_vectorization(file_path):
    loaded_model = tf.keras.models.load_model(file_path)
    return loaded_model.layers[
        0
    ]  # Lúc save thì cho vào dummy model, nên lúc lấy ra thì get layer đầu tiên


# Create dataset từ các câu english và spanish
def format_dataset(
    eng, spa, source_vectorization, target_vectorization, source_name, target_name
):
    eng = source_vectorization(eng)
    spa = target_vectorization(spa)
    return (
        {
            source_name: eng,  # English thì bình thường
            target_name: spa[:, :-1],  # Ko lấy token cuối cùng với Spanish
        },
        spa[:, 1:],  # Dịch chuyển 1 bước lên đối với Spanish
    )


# Create dataset từ list tuple (eng, spa)
def transform_dataset(
    ds,
    source_vectorization,
    target_vectorization,
    source_name,
    target_name,
):

    ds_transformed = ds.map(
        lambda eng, spa: format_dataset(
            eng,
            spa,
            source_vectorization,
            target_vectorization,
            source_name,
            target_name,
        ),
        num_parallel_calls=4,
    )

    # Prefetch và cache
    ds_transformed = tf_myfuncs.cache_prefetch_tfdataset_2(ds_transformed)

    return ds_transformed
