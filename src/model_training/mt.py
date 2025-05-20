import tensorflow as tf
import time
from Mylib import tf_myfuncs, myfuncs
from src.utils import classes, funcs


def load_data(data_transformation_path):
    train_ds = tf.data.Dataset.load(f"{data_transformation_path}/train_ds")
    val_ds = tf.data.Dataset.load(f"{data_transformation_path}/val_ds")

    return train_ds, val_ds


def create_GRU_model_from_layers(model):
    # Tạo phần encoder
    encoder = model["encoder"]

    source = encoder[0]
    x = source

    for layer in encoder[1:-1]:
        x = layer(x)

    encoded_source = encoder[-1](x)

    # Tạo phần decoder
    ## Tạo phần đầu
    decoder = model["decoder"]
    decoder_start = decoder["start"]
    past_target = decoder_start[0]
    x = past_target

    for layer in decoder_start[1:]:
        x = layer(x)

    # Tạo phần giữa
    decoder_gru = decoder["mid"]
    x = decoder_gru(x, initial_state=encoded_source)

    # Tạo phần cuối
    decoder_end = decoder["end"]

    for layer in decoder_end[:-1]:
        x = layer(x)

    target_next_step = decoder_end[-1](x)

    # Tạo model
    seq2seq_rnn = tf.keras.Model(inputs=[source, past_target], outputs=target_next_step)

    return seq2seq_rnn


def create_and_save_GRU_models_before_training(
    model_training_path,
    model_indices,
    models,
):
    for model_index, model in zip(model_indices, models):
        model_path = f"{model_training_path}/{model_index}.keras"
        model = create_GRU_model_from_layers(model)

        # Save model
        model.save(model_path)


def create_callbacks(
    callbacks,
    model_path,
    best_model_scoring_path,
    target_score,
    model_checkpoint_monitor,
):
    callbacks = [tf_myfuncs.copy_one_callback(callback) for callback in callbacks]

    callbacks = [
        classes.CustomisedModelCheckpoint(
            filepath=model_path,
            scoring_path=best_model_scoring_path,
            monitor=model_checkpoint_monitor,
            indicator=target_score,
        ),
    ] + callbacks

    return callbacks


def get_train_val_scoring_to_plot(train_scoring, val_scoring, scoring):
    if scoring == "accuracy":
        train_scoring, val_scoring = train_scoring * 100, val_scoring * 100

    return train_scoring, val_scoring


def train_and_save_GRU_models(
    model_training_path,
    model_indices,
    train_ds,
    val_ds,
    epochs,
    callbacks,
    model_name,
    target_score,
    scoring,
    plot_dir,
    optimizer,
    loss,
):
    tf.config.run_functions_eagerly(True)  # Bật eager execution
    tf.data.experimental.enable_debug_mode()  # Bật chế độ eager cho tf.data

    # TODO: d
    print("Update lần 1")
    # d

    print(f"===========Bắt đầu train model name = {model_name}===============\n")
    start_time = time.time()  # Bắt đầu tính thời gian train model
    for model_index in model_indices:
        # Load model
        model_path = f"{model_training_path}/{model_index}.keras"
        model = tf.keras.models.load_model(model_path)

        # Đường dẫn để lưu scoring ứng với best model
        best_model_scoring_path = f"{model_training_path}/{model_index}_scoring.pkl"

        # Create optimizer cho model
        model_optimizer = tf_myfuncs.copy_one_optimizer(optimizer)

        # Compile model trước khi training
        model.compile(
            optimizer=model_optimizer,
            loss=loss,
            metrics=[funcs.get_metric_object_by_name(scoring)],
        )

        # Create callbacks cho model
        model_callbacks = create_callbacks(
            callbacks,
            model_path,
            best_model_scoring_path,
            target_score,
            scoring,
        )

        # Train model
        print(f"Bắt đầu train model index {model_name} - {model_index}")
        history = model.fit(
            train_ds,
            epochs=epochs,
            verbose=1,
            validation_data=val_ds,
            callbacks=model_callbacks,
        )
        print(f"Kết thúc train model index {model_name} - {model_index}")
        num_epochs_before_stopping = len(history.history["loss"])

        # Đánh giá model
        train_scoring, val_scoring = myfuncs.load_python_object(best_model_scoring_path)

        ## In kết quả
        print("Kết quả của model")
        print(
            f"Model index {model_name} - {model_index}\n -> Train {scoring}: {train_scoring}, Val {scoring}: {val_scoring}, Epochs: {num_epochs_before_stopping}/{epochs}\n"
        )

        # Lưu dữ liệu để vẽ biểu đồ
        model_name_in_plot = f"{model_name}_{model_index}"

        train_scoring, val_scoring = get_train_val_scoring_to_plot(
            train_scoring, val_scoring, scoring
        )

        myfuncs.save_python_object(
            f"{plot_dir}/{model_name_in_plot}.pkl",
            (model_name_in_plot, train_scoring, val_scoring),
        )

    all_model_end_time = time.time()  # Kết thúc tính thời gian train model
    true_all_models_train_time = (all_model_end_time - start_time) / 60

    print(f"Thời gian chạy tất cả: {true_all_models_train_time} (mins)")

    print(f"===========Kết thúc train  model name = {model_name}===============\n")
