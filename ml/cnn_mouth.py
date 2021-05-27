import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from datetime import datetime

import std_dataset as d
from seaborn_plot import plot_history


model_saving_dir = Path("../models/mouth/")

tensorboard_log_dir = Path("../tensorboard/")


def get_model_name(label="-mouth"):
    utc_now = datetime.utcnow()
    return f"{utc_now.strftime('%y-%m-%d_%H-%M-%S')}{label}.h5"


def create_model(input_shape) -> keras.Sequential:
    return keras.Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        # layers.Conv2D(64, 3, padding='same', activation='relu'),
        # layers.MaxPooling2D(),
        # layers.Conv2D(64, 3, padding='same', activation='relu'),
        # layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(560, activation='relu',
                     bias_regularizer=keras.regularizers.L2(0.01),
                     activity_regularizer=keras.regularizers.L2(0.01)),
        # layers.Dropout(0.5),
        layers.Dense(890, activation='relu',
                     activity_regularizer=keras.regularizers.L2(0.01)),
        # layers.Dropout(0.5),
        layers.Dense(890, activation='relu'),
        layers.Dense(890, activation='relu'),
        layers.Dense(890, activation='relu'),
        layers.Dense(890, activation='relu'),
        layers.Dense(890, activation='relu'),
        layers.Dense(890, activation='relu'),
        # layers.Dropout(0.25),
        layers.Dense(700, activation='relu'),
        layers.Dense(6),
    ], 'cnn-mdl')


test_set_proportion = 0.1
val_set_proportion = 0.2  # Proportion of validation set to train set

epoch_count = 50


def main():
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Support GPU acceleration? {tf.test.is_gpu_available()}")

    print("Fetching dataset...")

    data_x, data_y = d.read_dataset("mouth")

    count = len(data_y)
    split = int(count * (1 - test_set_proportion))

    train_x, train_y = data_x[:split], data_y[:split]
    test_x, test_y = data_x[split:], data_y[split:]

    print('Fetching dataset done')

    print("Creating model... ", end='')
    model = create_model((80, 50, 3))
    print('done')

    print("Configuring model... ", end='')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    print("done")

    print("Initializing TensorBoard callback... ", end="")
    tb_callback = keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
    print("done")

    print("Training...")
    history = model.fit(x=train_x, y=train_y,
                        epochs=epoch_count,
                        validation_split=val_set_proportion,
                        batch_size=6,
                        callbacks=[tb_callback])
    print("Training done")

    print("Evaluating...")
    loss, acc = model.evaluate(test_x, test_y)
    print("Evaluating done")

    print()
    print("Test set:")
    print(f"\tLoss: {loss}")
    print(f"\tAccuracy: {acc}")
    print()

    print("Saving model... ", end='')
    model.save(model_saving_dir / get_model_name())
    print('done')

    print("Plotting data... ", end='')
    plot_history(history, epoch_count, 5)
    print('done')


if __name__ == '__main__':
    main()
