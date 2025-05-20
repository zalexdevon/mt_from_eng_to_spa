import tensorflow as tf
from src.utils import classes


def load_data(data_transformation_path, model_path):
    train_ds = tf.data.Dataset.load(f"{data_transformation_path}/train_ds")
    val_ds = tf.data.Dataset.load(f"{data_transformation_path}/val_ds")
    model = tf.keras.models.load_model(model_path)

    return train_ds, val_ds, model


def evaluate_model_on_train_val(
    train_ds, val_ds, model, model_evaluation_on_train_val_path
):
    final_model_results_text = (
        "===============Kết quả đánh giá model==================\n"
    )

    # Đánh giá model trên tập train, val
    model_results_text = classes.MachineTranslationEvaluator(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
    ).evaluate()

    final_model_results_text += model_results_text

    # Ghi vào file result.txt
    with open(f"{model_evaluation_on_train_val_path}/result.txt", mode="w") as file:
        file.write(final_model_results_text)
