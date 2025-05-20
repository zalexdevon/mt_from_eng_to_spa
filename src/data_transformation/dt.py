from Mylib import myfuncs
import tensorflow as tf


def load_data(train_pairs_path, val_pairs_path):
    train_pairs = myfuncs.load_python_object(train_pairs_path)
    val_pairs = myfuncs.load_python_object(val_pairs_path)

    return train_pairs, val_pairs


def create_vocabularies(train_pairs, source_vectorization, target_vectorization):
    train_english_texts = [pair[0] for pair in train_pairs]
    source_vectorization.adapt(train_english_texts)

    train_spanish_texts = [pair[1] for pair in train_pairs]
    target_vectorization.adapt(train_spanish_texts)


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
def make_dataset(
    pairs,
    batch_size,
    source_vectorization,
    target_vectorization,
    source_name,
    target_name,
):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)

    # Tạo tf.data.Dataset và chia batch
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)

    # Convert về đúng định dạng
    dataset = dataset.map(
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
    dataset = dataset.shuffle(2048).prefetch(16).cache()
    return dataset


def create_and_save_datasets(
    data_transformation_path,
    source_vectorization,
    target_vectorization,
    train_pairs,
    val_pairs,
    batch_size,
    source_name,
    target_name,
):
    train_ds = make_dataset(
        train_pairs,
        batch_size,
        source_vectorization,
        target_vectorization,
        source_name,
        target_name,
    )
    val_ds = make_dataset(
        val_pairs,
        batch_size,
        source_vectorization,
        target_vectorization,
        source_name,
        target_name,
    )

    # Save data
    train_ds.save(f"{data_transformation_path}/train_ds")
    val_ds.save(f"{data_transformation_path}/val_ds")
