from Mylib import myfuncs, tf_myfuncs
import tensorflow as tf
from src.utils import funcs


def load_data(train_ds_path, val_ds_path):
    train_ds = tf.data.Datset.load(train_ds_path)
    val_ds = tf.data.Datset.load(val_ds_path)

    return train_ds, val_ds


def create_vocabularies(train_ds, source_vectorization, target_vectorization):
    train_english_texts = train_ds.map(lambda x, y: x)
    source_vectorization.adapt(train_english_texts)

    train_spanish_texts = train_ds.map(lambda x, y: y)
    target_vectorization.adapt(train_spanish_texts)


def create_and_save_data(
    data_transformation_path,
    source_vectorization,
    target_vectorization,
    train_ds,
    val_ds,
    source_name,
    target_name,
):
    train_ds = funcs.transform_dataset(
        train_ds, source_vectorization, target_vectorization, source_name, target_name
    )
    val_ds = funcs.transform_dataset(
        val_ds, source_vectorization, target_vectorization, source_name, target_name
    )

    # Save data
    train_ds.save(f"{data_transformation_path}/train_ds")
    val_ds.save(f"{data_transformation_path}/val_ds")
    funcs.save_text_vectorization(
        source_vectorization, f"{data_transformation_path}/source_vectorization.keras"
    )
    funcs.save_text_vectorization(
        target_vectorization, f"{data_transformation_path}/target_vectorization.keras"
    )
