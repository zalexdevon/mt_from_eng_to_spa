import tensorflow as tf


def convert_list_tuples_to_dataset(pairs, batch_size):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)

    # Tạo tf.data.Dataset và chia batch
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)

    return dataset


def create_3_datasets(train_pairs, val_pairs, test_pairs, batch_size):
    train_ds = convert_list_tuples_to_dataset(train_pairs, batch_size)
    val_ds = convert_list_tuples_to_dataset(val_pairs, batch_size)
    test_ds = convert_list_tuples_to_dataset(test_pairs, batch_size)

    return train_ds, val_ds, test_ds
