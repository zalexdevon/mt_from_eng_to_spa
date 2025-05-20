from src.utils import funcs, classes
import tensorflow as tf


def load_data(test_ds_path, data_transformation_path, model_path):
    test_ds = tf.data.Dataset.load(test_ds_path)
    source_vectorization = funcs.load_text_vectorization(
        f"{data_transformation_path}/source_vectorization.keras"
    )
    target_vectorization = funcs.load_text_vectorization(
        f"{data_transformation_path}/target_vectorization.keras"
    )

    model = tf.keras.models.load_model(model_path)

    return test_ds, source_vectorization, target_vectorization, model


def evaluate_and_save_data(test_ds, model, model_evaluation_on_test_path):
    final_model_results_text = (
        "===============Kết quả đánh giá model==================\n"
    )

    # Đánh giá model trên tập train, val
    model_results_text = classes.MachineTranslationEvaluator(
        model=model,
        train_ds=test_ds,
    ).evaluate()

    final_model_results_text += model_results_text

    # Ghi vào file result.txt
    with open(f"{model_evaluation_on_test_path}/result.txt", mode="w") as file:
        file.write(final_model_results_text)
